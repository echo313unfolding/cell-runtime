#!/usr/bin/env python3
"""almanac — CLI for the Almanac evidence pipeline.

Commands:
  almanac normalize <takeout-path> [--member-id ID] [--output PATH]
  almanac vault store   [--evidence-path PATH]
  almanac vault approve [--evidence-path PATH]
  almanac vault commit  [--evidence-path PATH]
  almanac vault verify  --evidence-hash HASH --receipt-hash HASH
  almanac vault status

The normalize step parses a Google Takeout export into a member evidence
table. The vault commands manage encrypted local storage with member
approval gates.

Requires:
  - ALMANAC_VAULT_SECRET env var or ~/.almanac/vault_secret
  - ~/almanac-agent/src/normalizer.py (for normalize command)
"""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

EVIDENCE_DIR = Path(os.path.expanduser("~/receipts/almanac"))
DEFAULT_EVIDENCE = EVIDENCE_DIR / "latest_evidence_table.json"
NORMALIZER_PATH = Path(os.path.expanduser("~/almanac-agent/src/normalizer.py"))


def _import_normalizer():
    """Import the normalizer from almanac-agent by path."""
    if not NORMALIZER_PATH.exists():
        print(f"error: normalizer not found at {NORMALIZER_PATH}", file=sys.stderr)
        print("       install almanac-agent or set ALMANAC_NORMALIZER_PATH", file=sys.stderr)
        sys.exit(1)
    norm_dir = str(NORMALIZER_PATH.parent)
    if norm_dir not in sys.path:
        sys.path.insert(0, norm_dir)
    import normalizer
    return normalizer


# ── normalize ────────────────────────────────────────────────────────────────

def cmd_normalize(args):
    """Parse Takeout export into evidence table."""
    normalizer = _import_normalizer()

    takeout_path = Path(args.takeout_path)
    if not takeout_path.exists():
        print(f"error: file not found: {takeout_path}", file=sys.stderr)
        sys.exit(1)

    member_id = args.member_id
    print(f"Normalizing: {takeout_path}")
    print(f"Member ID:   {member_id}")

    t0 = time.time()
    table = normalizer.normalize_youtube_takeout(str(takeout_path), member_id=member_id)
    elapsed = round(time.time() - t0, 3)

    summary = table.summary()
    print(f"\nParsed {summary['total_records']} records in {elapsed}s")
    print(f"  Mapped:     {summary['mapped']}")
    print(f"  Unmapped:   {summary['unmapped']}")
    print(f"  Invalid:    {summary['invalid']}")
    print(f"  Attributes: {summary['attributes_extracted']}")
    print(f"  Provenance: {summary['provenance_coverage']}")

    # Signal quality breakdown
    sq = summary.get("signal_quality_distribution", {})
    if sq:
        print(f"\n  Signal quality:")
        for bucket, count in sq.items():
            print(f"    {bucket}: {count}")

    # Save evidence table
    output_path = Path(args.output) if args.output else DEFAULT_EVIDENCE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save aggregate attributes (channel affinities + distributions)
    agg_attrs = [a for a in table.attributes
                 if a.attribute_name in ("interest_channel_affinity",
                                         "signal_quality_distribution",
                                         "activity_recency")]
    table_dict = asdict(table)
    table_dict["attributes"] = [asdict(a) for a in agg_attrs]
    table_dict["per_video_attributes_count"] = len(table.attributes) - len(agg_attrs)

    output_path.write_text(json.dumps(table_dict, indent=2, default=str))
    print(f"\nEvidence table: {output_path}")
    print(f"  Source SHA256: {table.source_file_sha256[:24]}...")
    print(f"  Member hash:   {table.member_hash}")


# ── vault store ──────────────────────────────────────────────────────────────

def cmd_vault_store(args):
    """Review evidence and show hash + summary (writes nothing)."""
    from cell.agents.vault_agent import VaultReviewAgent

    agent = VaultReviewAgent()
    params = {"action": "store"}
    if args.evidence_path:
        params["evidence_path"] = args.evidence_path

    result = agent.execute(params)
    if not result.ok:
        print(f"error: {result.error}", file=sys.stderr)
        sys.exit(1)

    out = result.output
    print(f"Evidence review:")
    print(f"  Hash:      {out['evidence_hash']}")
    print(f"  Source:    {out['evidence_source']}")
    print(f"  Member:    {out['member_hash']}")
    print(f"  Records:   {out['total_records']}")
    print(f"  Version:   {out['transform_version']}")

    attr_types = out.get("attribute_types", {})
    if attr_types:
        print(f"  Attributes:")
        for name, count in sorted(attr_types.items()):
            print(f"    {name}: {count}")

    print(f"\nTo approve and commit:")
    print(f"  almanac vault approve --evidence-path {out['evidence_source']}")
    print(f"  almanac vault commit  --evidence-path {out['evidence_source']}")


# ── vault approve ────────────────────────────────────────────────────────────

def cmd_vault_approve(args):
    """Interactive member approval (reads from stdin)."""
    from cell.agents.vault_agent import VaultReviewAgent, _load_evidence, _get_vault_secret, VAULT_DIR
    from cell.vault.approval import prompt_member_approval

    # Load evidence to get hash and member
    try:
        path, evidence_data, _, evidence_hash = _load_evidence(args.evidence_path)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    member_hash = evidence_data["member_hash"]

    secret = _get_vault_secret()
    if not secret:
        print("error: no vault secret. Set ALMANAC_VAULT_SECRET or create ~/.almanac/vault_secret",
              file=sys.stderr)
        sys.exit(1)

    # Show review summary first
    agent = VaultReviewAgent()
    params = {"action": "store"}
    if args.evidence_path:
        params["evidence_path"] = args.evidence_path
    review = agent.execute(params)
    if not review.ok:
        print(f"error: {review.error}", file=sys.stderr)
        sys.exit(1)

    evidence_summary = {
        "total_records": review.output["total_records"],
        "transform_version": review.output["transform_version"],
        "attribute_types": review.output.get("attribute_types", {}),
    }

    # Interactive approval prompt
    approval_dir = VAULT_DIR / "approvals"
    approval = prompt_member_approval(
        approval_dir=approval_dir,
        evidence_hash=evidence_hash,
        member_hash=member_hash,
        scope="storage_only",
        vault_secret=secret,
        evidence_summary=evidence_summary,
    )

    if approval is None:
        print("Denied.", file=sys.stderr)
        sys.exit(1)

    print(f"\nApproval created:")
    print(f"  ID:      {approval['approval_id']}")
    print(f"  Scope:   {approval['scope']}")
    print(f"  Expires: {approval['expires_at']}")
    print(f"\nNow run within 5 minutes:")
    ep = f" --evidence-path {args.evidence_path}" if args.evidence_path else ""
    print(f"  almanac vault commit{ep}")


# ── vault commit ─────────────────────────────────────────────────────────────

def cmd_vault_commit(args):
    """Commit approved evidence to encrypted vault."""
    from cell.agents.vault_agent import VaultReviewAgent, _load_evidence

    # Need the evidence hash for commit
    try:
        _, _, _, evidence_hash = _load_evidence(args.evidence_path)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    agent = VaultReviewAgent()
    params = {
        "action": "commit",
        "evidence_hash": evidence_hash,
        "approval_scope": "storage_only",
    }
    if args.evidence_path:
        params["evidence_path"] = args.evidence_path

    result = agent.execute(params)
    if not result.ok:
        print(f"error: {result.error}", file=sys.stderr)
        sys.exit(1)

    out = result.output
    print(f"Committed to vault:")
    print(f"  Evidence hash:  {out['evidence_hash']}")
    print(f"  Receipt hash:   {out['receipt_hash']}")
    print(f"  Receipt ID:     {out['receipt_id']}")
    print(f"  Chain position: {out['chain_position']}")
    print(f"  Vault:          {out['vault_dir']}")
    print(f"  Transaction:    {out['transaction_path']}")
    print(f"  Time:           {out['wall_time_s']}s")
    print(f"\nTo verify:")
    print(f"  almanac vault verify --evidence-hash {out['evidence_hash']} --receipt-hash {out['receipt_hash']}")


# ── vault verify ─────────────────────────────────────────────────────────────

def cmd_vault_verify(args):
    """Decrypt and verify a vault record."""
    from cell.agents.vault_agent import VaultReviewAgent

    if not args.evidence_hash or not args.receipt_hash:
        print("error: --evidence-hash and --receipt-hash are required", file=sys.stderr)
        sys.exit(1)

    agent = VaultReviewAgent()
    result = agent.execute({
        "action": "verify",
        "evidence_hash": args.evidence_hash,
        "receipt_hash": args.receipt_hash,
    })

    if not result.ok:
        print(f"error: {result.error}", file=sys.stderr)
        sys.exit(1)

    out = result.output
    bs = out.get("bundle_summary", {})
    print(f"Verified:")
    print(f"  Status:       {out['status']}")
    print(f"  Evidence:     {out['evidence_hash']}")
    print(f"  Receipt:      {out['receipt_hash']}")
    print(f"  Member:       {out['member_hash']}")
    print(f"  Schema:       {bs.get('schema', '?')}")
    print(f"  Records:      {bs.get('total_records', '?')}")
    print(f"  Attributes:   {bs.get('attribute_count', '?')}")
    print(f"  Verify time:  {out.get('verification_time_s', '?')}s")


# ── vault status ─────────────────────────────────────────────────────────────

def cmd_vault_status(args):
    """Show vault state."""
    from cell.agents.vault_agent import VaultReviewAgent

    agent = VaultReviewAgent()
    result = agent.execute({"action": "status"})
    if not result.ok:
        print(f"error: {result.error}", file=sys.stderr)
        sys.exit(1)

    out = result.output
    print(f"Vault status:")
    for key, val in sorted(out.items()):
        print(f"  {key}: {val}")


# ── Argument parser ──────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        prog="almanac",
        description="Almanac evidence pipeline — normalize, vault, verify.",
    )
    sub = parser.add_subparsers(dest="command")

    # normalize
    norm = sub.add_parser("normalize", help="Parse Takeout export into evidence table")
    norm.add_argument("takeout_path", help="Path to watch-history.html")
    norm.add_argument("--member-id", default="member_0",
                      help="Member identifier (default: member_0)")
    norm.add_argument("--output", "-o", default=None,
                      help=f"Output path (default: {DEFAULT_EVIDENCE})")
    norm.set_defaults(func=cmd_normalize)

    # vault (subcommand group)
    vault = sub.add_parser("vault", help="Encrypted evidence vault operations")
    vault_sub = vault.add_subparsers(dest="vault_command")

    # vault store
    vs = vault_sub.add_parser("store", help="Review evidence (writes nothing)")
    vs.add_argument("--evidence-path", default=None,
                    help="Path to evidence table JSON")
    vs.set_defaults(func=cmd_vault_store)

    # vault approve
    va = vault_sub.add_parser("approve", help="Interactive member approval")
    va.add_argument("--evidence-path", default=None,
                    help="Path to evidence table JSON")
    va.set_defaults(func=cmd_vault_approve)

    # vault commit
    vc = vault_sub.add_parser("commit", help="Commit approved evidence to vault")
    vc.add_argument("--evidence-path", default=None,
                    help="Path to evidence table JSON")
    vc.set_defaults(func=cmd_vault_commit)

    # vault verify
    vv = vault_sub.add_parser("verify", help="Decrypt and verify vault record")
    vv.add_argument("--evidence-hash", required=True)
    vv.add_argument("--receipt-hash", required=True)
    vv.set_defaults(func=cmd_vault_verify)

    # vault status
    vst = vault_sub.add_parser("status", help="Show vault state")
    vst.set_defaults(func=cmd_vault_status)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "vault" and not getattr(args, "vault_command", None):
        # Show vault subcommand help
        parser.parse_args(["vault", "--help"])
        sys.exit(0)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
