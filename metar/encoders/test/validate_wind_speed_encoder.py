from encoders.wind_speed_encoder import (
    WindSpeedEncoder,
)

from .scalar_encoder_validation import (
    ScalarEncoderValidationSuite,
)


def validate_global_range():

    encoder = WindSpeedEncoder()

    validator = ScalarEncoderValidationSuite(
        encoder=encoder,
        variable_name="wind_speed_global",
        value_range=(0, 150),
    )

    validator.run_all()


def validate_low_speed():

    encoder = WindSpeedEncoder()

    validator = ScalarEncoderValidationSuite(
        encoder=encoder,
        variable_name="wind_speed_low_speed",
        value_range=(0, 20),
    )

    validator.run_all()


def validate_operational_range():

    encoder = WindSpeedEncoder()

    validator = ScalarEncoderValidationSuite(
        encoder=encoder,
        variable_name="wind_speed_operational",
        value_range=(0, 60),
    )

    validator.run_all()


def validate_storm_regime():

    encoder = WindSpeedEncoder()

    validator = ScalarEncoderValidationSuite(
        encoder=encoder,
        variable_name="wind_speed_storm",
        value_range=(20, 100),
    )

    validator.run_all()


def main():

    print()
    print("=" * 80)
    print("VALIDATING WIND SPEED ENCODER")
    print("=" * 80)
    print()

    validate_global_range()

    validate_low_speed()

    validate_operational_range()

    validate_storm_regime()

    print()
    print("=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
