import argparse
import json

from audit_core import parse_rules, rules_to_structured_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TXT rules to structured JSON.")
    parser.add_argument("--input", required=True, help="Rules TXT file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as handle:
        raw_text = handle.read()

    rules = parse_rules(raw_text)
    payload = rules_to_structured_payload(rules)

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {len(payload)} rules to {args.output}")


if __name__ == "__main__":
    main()
