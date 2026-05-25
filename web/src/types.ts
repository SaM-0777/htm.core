export interface Encoded {
  value_encoded: number | string | null | undefined;
  size: number;
  active_bits: number[];
  active_count: number;
}

export interface EncodedValues {
  //status: "success" | "error";
  encoders: {
    recorded_time: Encoded;
    temperature: Encoded;
    pressure: Encoded;
    dew_point: Encoded;
    visibility: Encoded;
    wind: Encoded;
    clouds: Encoded;
  };
}

export interface CloudLayer {
  coverage: string;
  height_ft: number;
  type: string;
}

export interface TimeState {
  year: number | "";
  month: number | "";
  day: number | "";
  hour: number | "";
  minute: number | "";
}

export interface ScalarState {
  value: number;
}

export interface WindState {
  speed: number;
  direction: number;
  isVariable: boolean;
  gust: number | null;
}
