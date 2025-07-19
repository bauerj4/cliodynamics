from crisiswatch_agent.agent import create_agent
from crisiswatch_agent.tools.fetch import init_db
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["openai", "smollm"], default="openai")
    args = parser.parse_args()

    init_db()
    agent = create_agent(args.model)
    agent.cli()
