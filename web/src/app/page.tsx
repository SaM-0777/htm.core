"use client";
import { useState } from "react";
import {
  CloudLayer,
  EncodedValues,
  ScalarState,
  TimeState,
  WindState,
} from "@/types";
import { Input } from "@/components/Inputs";
import { Slider } from "@/components/Slider";
import { Toggle } from "@/components/Toggle";
import { Select } from "@/components/Select";

export default function Page() {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [time, setTime] = useState<TimeState>(() => {
    const date = new Date();
    const year = date.getUTCFullYear();
    const month = date.getUTCMonth() + 1;
    const day = date.getUTCDate();
    const hour = date.getUTCHours();
    const minute = date.getUTCMinutes();

    return {
      year,
      month,
      day,
      hour,
      minute,
    };
  });
  const [wind, setWind] = useState<WindState>({
    speed: 12,
    direction: 180,
    isVariable: false,
    gust: null,
  });
  const [clouds, setClouds] = useState<CloudLayer[]>([]);
  const [temperature, setTemperature] = useState<ScalarState>({ value: 22 });
  const [pressure, setPressure] = useState<ScalarState>({ value: 1015 });
  const [dewPoint, setDewPoint] = useState<ScalarState>({ value: 12 });
  const [visibility, setVisibility] = useState<ScalarState>({ value: 10000 });

  const clearSDR = () => {
    //setEncodedValues(null)
  };
  const handleTime = <K extends keyof TimeState>(
    field: K,
    val: TimeState[K],
  ) => {
    setTime((prev) => ({ ...prev, [field]: val }));
    clearSDR();
  };
  const handleWind = <K extends keyof WindState>(
    field: K,
    val: WindState[K],
  ) => {
    setWind((prev) => ({ ...prev, [field]: val }));
    clearSDR();
  };
  const addCloudLayer = () => {
    if (clouds.length >= 3) return;
    setClouds((prev) => [
      ...prev,
      { coverage: "SCT", height_ft: 2500, type: "NONE" },
    ]);
    clearSDR();
  };
  const updateCloudLayer = <K extends keyof CloudLayer>(
    index: number,
    field: K,
    val: CloudLayer[K],
  ) => {
    setClouds((prev) =>
      prev.map((layer, i) =>
        i === index ? { ...layer, [field]: val } : layer,
      ),
    );
    clearSDR();
  };
  const removeCloudLayer = (index: number) => {
    setClouds((prev) => prev.filter((_, i) => i !== index));
    clearSDR();
  };
  const handleTemperature = (value: number) => {
    setTemperature({ value });
    clearSDR();
  };
  const handlePressure = (value: number) => {
    setPressure({ value });
    clearSDR();
  };
  const handleDewPoint = (value: number) => {
    setDewPoint({ value });
    clearSDR();
  };
  const handleVisibility = (value: number) => {
    setVisibility({ value });
    clearSDR();
  };

  const [encodedValues, setEncodedValues] = useState<EncodedValues | null>(
    null,
  );

  async function encode() {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL!;

    if (!apiUrl) {
      return;
    }

    setIsLoading(true);
    try {
      const url = `${apiUrl}/api/v1/encode`;

      const y = time.year || 2026;
      const m = time.month || 1;
      const d = time.day || 1;
      const h = time.hour || 0;
      const mn = time.minute || 0;

      const formattedTime = `${y}-${String(m).padStart(2, "0")}-${String(d).padStart(2, "0")}T${String(h).padStart(2, "0")}:${String(mn).padStart(2, "0")}:00Z`;

      const body = {
        time_recorded: formattedTime,
        temperature_c: temperature.value,
        pressure_hpa: pressure.value,
        dew_point_c: dewPoint.value,
        visibility: visibility.value,
        wind_speed_kt: wind.speed,
        wind_direction_deg: wind.direction,
        is_wind_variable: wind.isVariable,
        wind_gust_kt: wind.gust,
        cloud_layers: clouds.map((c: CloudLayer) => ({
          coverage: c.coverage,
          height_ft: c.height_ft,
          type: c.type === "NONE" ? null : c.type,
        })),
      };
      const req = await fetch(url, {
        method: "POST",
        body: JSON.stringify(body),
        headers: {
          "Content-Type": "application/json",
        },
      });

      const res = (await req.json()) as { data: EncodedValues };
      console.log({ res });
      setEncodedValues(res.data);
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
    } catch (error) {
      return;
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="min-h-screen text-slate-200 font-sans selection:bg-white/20 overflow-hidden flex flex-col md:flex-row">
      {/* Left Panel */}
      <div className="w-full md:w-100 shrink-0 z-10 flex flex-col justify-between gap-12 border-b md:border-b-0 md:border-r border-white/5 bg-black/40 backdrop-blur-xl p-6 h-auto md:h-screen overflow-y-auto">
        <div className="flex flex-col items-start gap-12 overflow-hidden">
          <h1 className="text-xs uppercase tracking-[0.3em] font-semibold text-white/80">
            Metar<span className="text-white">Visualizer</span>
          </h1>

          <div className="flex flex-col items-start gap-y-12 h-full overflow-y-auto no-scrollbar">
            <div className="space-y-6">
              <h2 className="text-[9px] uppercase tracking-widest text-slate-600 border-b border-white/10 pb-2">
                Observation Time (UTC)
              </h2>
              <div className="flex gap-3 justify-between items-end">
                <div className="flex flex-col gap-1 items-center">
                  <span className="text-[7px] text-slate-500 uppercase tracking-widest">
                    YYYY
                  </span>
                  <Input
                    value={time.year}
                    onChange={(v) => handleTime("year", v)}
                    min={1900}
                    max={2100}
                    className="w-14"
                  />
                </div>
                <span className="text-white/30 pb-1">-</span>
                <div className="flex flex-col gap-1 items-center">
                  <span className="text-[7px] text-slate-500 uppercase tracking-widest">
                    MM
                  </span>
                  <Input
                    value={time.month}
                    onChange={(v) => handleTime("month", v)}
                    min={1}
                    max={12}
                    className="w-10"
                  />
                </div>
                <span className="text-white/30 pb-1">-</span>
                <div className="flex flex-col gap-1 items-center">
                  <span className="text-[7px] text-slate-500 uppercase tracking-widest">
                    DD
                  </span>
                  <Input
                    value={time.day}
                    onChange={(v) => handleTime("day", v)}
                    min={1}
                    max={31}
                    className="w-10"
                  />
                </div>
                <span className="text-white/30 pb-1 ml-2">:</span>
                <div className="flex flex-col gap-1 items-center">
                  <span className="text-[7px] text-slate-500 uppercase tracking-widest">
                    HH
                  </span>
                  <Input
                    value={time.hour}
                    onChange={(v) => handleTime("hour", v)}
                    min={0}
                    max={23}
                    className="w-10"
                  />
                </div>
                <span className="text-white/30 pb-1">:</span>
                <div className="flex flex-col gap-1 items-center">
                  <span className="text-[7px] text-slate-500 uppercase tracking-widest">
                    mm
                  </span>
                  <Input
                    value={time.minute}
                    onChange={(v) => handleTime("minute", v)}
                    min={0}
                    max={59}
                    className="w-10"
                  />
                </div>
              </div>
            </div>

            <div className="w-full space-y-6">
              <h2 className="text-[9px] uppercase tracking-widest text-slate-600 border-b border-white/10 pb-2">
                Atmospheric
              </h2>
              <Slider
                label="Temperature"
                value={temperature.value}
                min={-30}
                max={50}
                unit="°C"
                onChange={(v) => handleTemperature(v)}
              />
              <Slider
                label="Dew Point"
                value={dewPoint.value}
                min={-30}
                max={50}
                unit="°C"
                onChange={(v) => handleDewPoint(v)}
              />
              <Slider
                label="Pressure"
                value={pressure.value}
                min={900}
                max={1050}
                unit="hPa"
                onChange={(v) => handlePressure(v)}
              />
              <Slider
                label="Visibility"
                value={visibility.value}
                min={0}
                max={10000}
                step={100}
                unit="m"
                onChange={(v) => handleVisibility(v)}
              />
            </div>

            <div className="w-full space-y-6">
              <h2 className="text-[9px] uppercase tracking-widest text-slate-600 border-b border-white/10 pb-2">
                Wind Vector
              </h2>

              <Slider
                label="Speed"
                value={wind.speed}
                min={0}
                max={150}
                unit="kt"
                onChange={(v) => handleWind("speed", v)}
              />

              <div className="space-y-3">
                <Slider
                  label="Direction"
                  value={wind.direction}
                  min={0}
                  max={360}
                  unit="°"
                  onChange={(v) => handleWind("direction", v)}
                />
                <Toggle
                  label="Variable Direction (VRB)"
                  checked={wind.isVariable}
                  onChange={(v) => handleWind("isVariable", v)}
                />
              </div>

              <div className="space-y-4 pt-4 border-t border-white/3">
                <Toggle
                  label="Enable Gusts"
                  checked={wind.gust !== null}
                  onChange={(chk) => handleWind("gust", chk ? 15 : null)}
                />
                {wind.gust !== null && (
                  <Slider
                    label="Gust Speed"
                    value={wind.gust}
                    min={0}
                    max={150}
                    unit="kt"
                    onChange={(v) => handleWind("gust", v)}
                  />
                )}
              </div>
            </div>

            <div className="w-full space-y-6">
              <div className="flex justify-between items-center border-b border-white/10 pb-2">
                <h2 className="text-[9px] uppercase tracking-widest text-slate-600">
                  Cloud Layers ({clouds.length}/3)
                </h2>
                <button
                  onClick={addCloudLayer}
                  disabled={clouds.length >= 3}
                  className="text-[12px] text-white/50 hover:text-white uppercase tracking-widest transition-colors disabled:opacity-20"
                  title="Add Cloud Layer"
                >
                  +
                </button>
              </div>

              {clouds.length === 0 && (
                <div className="text-[10px] text-slate-700 font-mono tracking-widest uppercase py-4">
                  No Cloud Layers (SKC/CLR)
                </div>
              )}

              {clouds.map((layer, idx) => (
                <div
                  key={idx}
                  className="p-5 border border-white/5 bg-white/1 space-y-6 relative group"
                >
                  <button
                    onClick={() => removeCloudLayer(idx)}
                    className="absolute top-3 right-3 text-white/30 hover:text-white opacity-0 group-hover:opacity-100 transition-opacity text-[10px]"
                  >
                    ✕
                  </button>
                  <div className="flex gap-4">
                    <Select
                      label="Coverage"
                      value={layer.coverage}
                      options={["CLR", "SKC", "FEW", "SCT", "BKN", "OVC", "VV"]}
                      onChange={(v) => updateCloudLayer(idx, "coverage", v)}
                    />
                    <Select
                      label="Type"
                      value={layer.type}
                      options={[
                        "NONE",
                        "CLR",
                        "SKC",
                        "CI",
                        "CS",
                        "CC",
                        "AC",
                        "AS",
                        "SC",
                        "ST",
                        "NS",
                        "TCU",
                        "CB",
                      ]}
                      onChange={(v) => updateCloudLayer(idx, "type", v)}
                    />
                  </div>
                  <Slider
                    label="Altitude"
                    value={layer.height_ft}
                    min={0}
                    max={40000}
                    step={100}
                    unit="ft"
                    onChange={(v) => updateCloudLayer(idx, "height_ft", v)}
                  />
                </div>
              ))}
            </div>
          </div>
        </div>

        <div>
          <button
            onClick={encode}
            disabled={isLoading}
            className={`w-full py-4 px-6 rounded-sm text-[10px] uppercase tracking-[0.2em] font-bold transition-all duration-500 relative overflow-hidden group border cursor-pointer
              ${
                isLoading
                  ? "bg-transparent border-white/20 text-white/50"
                  : "bg-transparent text-white border-white/50 hover:border-white"
              }
            `}
          >
            <span
              className={`relative z-10 flex items-center justify-center gap-2 transition-colors duration-500`}
            >
              {isLoading ? "Processing Topology..." : "Encode to SDR"}
            </span>
          </button>
        </div>
      </div>

      {/* Right Panel */}
      <div className="flex-1 w-full border-l border-white/5 p-6 h-screen overflow-y-auto relative">
        <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-200 h-200 bg-white/5 rounded-full blur-[120px] pointer-events-none" />

        {!encodedValues ? (
          <div className="h-full flex items-center justify-center text-slate-700 text-[10px] uppercase tracking-widest">
            {isLoading
              ? "Encoding Neural Representation..."
              : "Enter parameters and encode to begin"}
          </div>
        ) : (
          <div className="w-full flex flex-col gap-6">
            {Object.entries(encodedValues.encoders).map(([key, data]) => {
              const activeSet = new Set(data.active_bits);

              return (
                <div
                  key={key}
                  className="w-full border border-white/10 p-6 bg-black/40"
                >
                  <div className="flex justify-between mb-4">
                    <h3 className="text-[10px] uppercase tracking-widest text-slate-300">
                      {key} SDR Space
                    </h3>
                    <span className="text-[9px] font-mono text-slate-500">
                      {data.active_count} / {data.size} bits
                    </span>
                  </div>
                  <div
                    className="grid gap-1 w-full"
                    style={{
                      gridTemplateColumns: `repeat(32, minmax(0, 1fr))`,
                    }}
                  >
                    {Array.from({ length: data.size }).map((_, i) => (
                      <div
                        key={i}
                        className={`w-full aspect-square rounded-[1px] ${activeSet.has(i) ? "bg-white shadow-[0_0_8px_rgba(255,255,255,0.6)]" : "bg-white/5"}`}
                      />
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </main>
  );
}
