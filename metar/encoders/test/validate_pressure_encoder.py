from encoders.pressure_encoder import (
    PressureEncoder,
)

from .scalar_encoder_validation import (
    validate_single_encoder,
)


def main():

    encoder = PressureEncoder()

    validate_single_encoder(
        encoder=encoder,
        variable_name="pressure",
        value_range=(870, 1085),
    )


if __name__ == "__main__":
    main()
