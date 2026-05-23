from encoders.wind_direction_encoder import (
    WindDirectionEncoder,
)

from .wind_direction_validation import (
    WindDirectionValidation,
)


def main():

    encoder = WindDirectionEncoder()

    validator = WindDirectionValidation(encoder)

    validator.run_all()


if __name__ == "__main__":
    main()
