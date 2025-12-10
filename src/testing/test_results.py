from src.utils.functions import ask_model_type_from_console
from src.testing.test import main

if __name__ == "__main__":
    model_type, white = ask_model_type_from_console()
    main()