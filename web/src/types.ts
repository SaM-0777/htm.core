export interface Encoded {
  value_encoded: number | string;
  size: number;
  active_bits: number[];
  active_count: number;
}

export interface EncodedValues {
  //status: "success" | "error";
  encoders: {
    temperature: Encoded;
    pressure: Encoded;
    dew_point: Encoded;
  };
}
