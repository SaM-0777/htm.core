export interface MetarValues {
  temperature: number;
  pressure: number;
  dewPoint: number;
  windSpeed: number;
}

export const metarParams = [
  {
    id: "temperature",
    label: "Temperature",
    min: -30,
    default: 22,
    max: 50,
    unit: "°C",
  },
  {
    id: "pressure",
    label: "Pressure",
    min: 950,
    default: 1015,
    max: 1050,
    unit: "hPa",
  },
  {
    id: "dewPoint",
    label: "Dew Point",
    min: -30,
    default: 14,
    max: 50,
    unit: "°C",
  },
  {
    id: "windSpeed",
    label: "Wind Speed",
    min: 0,
    default: 12,
    max: 100,
    unit: "kt",
  },
];
