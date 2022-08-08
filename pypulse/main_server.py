from main import main
from pathlib import Path

if __name__ == "__main__":
    root = Path().cwd() / "tickets"
    new_ticket_root = root / "test_new_structure"
    ticket = new_ticket_root / "pulsation.ini"

    main(ticket, run_laptop=False)