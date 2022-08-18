from main import main
from pathlib import Path

if __name__ == "__main__":


    root = Path().cwd() / "tickets"
    # ticket = folder / "2SPOTS_EQU_CLOSER_HIP73620_template.ini"

    ticket = root / "HIP16335_big_ticket.ini"

    main(ticket, run_laptop=False)
    exit()

    # exit()
    folder = root / "done_two_spots_diff_templates"
    tickets = folder.glob("*.ini")
    for ticket in tickets:
        main(ticket, run_laptop=False)


