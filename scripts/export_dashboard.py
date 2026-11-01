"""Copy model results and refresh every live file the dashboard reads."""
from hazards.refresh import cycle

if __name__ == "__main__":
    cycle(full=True)
