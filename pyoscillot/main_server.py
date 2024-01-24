from main import main
from pathlib import Path

if __name__ == "__main__":


    root = Path().cwd() / "tickets"

    folder = root / "over_vacation"
    tickets = sorted(list(folder.glob("*.ini")))
    for ticket in tickets:
        main(ticket, run_laptop=False)


