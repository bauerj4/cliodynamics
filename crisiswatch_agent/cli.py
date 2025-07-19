import argparse
from crisiswatch_agent.agent import create_agent


def main():
    parser = argparse.ArgumentParser(description="CrisisWatch Agent CLI")
    parser.add_argument(
        "--chat", action="store_true", help="Launch interactive chat mode"
    )
    parser.add_argument("--query", type=str, help="One-off query to run with the agent")
    parser.add_argument("--model", choices=["openai", "smollm"], default="smollm")

    args = parser.parse_args()

    agent = create_agent(model=args.model)

    if args.chat:
        print("\nInteractive Chat Mode (type 'exit' to quit)\n")
        while True:
            try:
                user_input = input("You: ")
                if user_input.strip().lower() in {"exit", "quit"}:
                    print("Goodbye!")
                    break
                response = agent.run(user_input)
                print(f"Agent: {response}\n")
            except (KeyboardInterrupt, EOFError):
                print("\nSession ended.")
                break

    elif args.query:
        print(agent.run(args.query))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
