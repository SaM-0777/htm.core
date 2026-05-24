"use client";
import { useState } from "react";
import { metarParams, MetarValues } from "@/constants/metar-params";
import { EncodedValues } from "@/types";

export default function Page() {
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [values, setValues] = useState<MetarValues>({
    temperature: 22,
    pressure: 1013,
    dewPoint: 14,
    windSpeed: 12,
  });
  const [encodedValues, setEncodedValues] = useState<EncodedValues | null>(
    null,
  );

  function handleSliderChange(id: keyof MetarValues, value: string) {
    setValues((prev) => ({ ...prev, [id]: parseFloat(value) }));
  }

  async function encode() {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL!;

    if (!apiUrl) {
      return;
    }

    setIsLoading(true);
    try {
      const url = `${apiUrl}/api/v1/encode`;
      const body = {
        temperature_c: values.temperature,
        pressure_hpa: values.pressure,
        dew_point_c: values.dewPoint,
        time_recorded: "2026-05-25T12:00:00Z",
      };
      const req = await fetch(url, {
        method: "POST",
        body: JSON.stringify(body),
        headers: {
          "Content-Type": "application/json",
        },
      });

      const res = (await req.json()) as { data: EncodedValues };
      setEncodedValues(res.data);
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
    } catch (error) {
      return;
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="min-h-screen text-slate-200 font-sans selection:bg-white/20 overflow-hidden flex flex-col md:flex-row relative">
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-200 h-200 bg-white/5 rounded-full blur-[120px] pointer-events-none" />

      {/* Left Panel */}
      <div className="w-full md:w-100 shrink-0 z-10 flex flex-col justify-between border-b md:border-b-0 md:border-r border-white/5 bg-black/40 backdrop-blur-xl p-8 md:p-12 h-auto md:h-screen overflow-y-auto">
        <div>
          <h1 className="text-xs uppercase tracking-[0.3em] font-semibold text-white/80">
            Metar<span className="text-white">Visualizer</span>
          </h1>

          <div className="space-y-12">
            {metarParams.map((p) => (
              <div key={p.label} className="group">
                <div className="flex justify-between items-baseline mb-4">
                  <label className="text-[10px] uppercase tracking-widest text-slate-500 font-medium group-hover:text-slate-300 transition-colors">
                    {p.label}
                  </label>
                  <span className="text-sm font-mono text-white/90">
                    {values[p.id as keyof MetarValues]}{" "}
                    <span className="text-slate-500 text-[10px] ml-1">
                      {p.unit}
                    </span>
                  </span>
                </div>

                <div className="relative h-0.5 bg-white/10 rounded-full flex items-center">
                  <input
                    type="range"
                    min={p.min}
                    max={p.max}
                    step={1}
                    value={values[p.id as keyof MetarValues]}
                    onChange={(e) =>
                      handleSliderChange(
                        p.id as keyof MetarValues,
                        e.target.value,
                      )
                    }
                    className="absolute w-full appearance-none bg-transparent outline-none z-20 h-4 cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(255,255,255,0.8)] transition-all"
                  />
                  <div
                    className={`absolute h-full rounded-full bg-white shadow-[0_0_8px_rgba(255,255,255,0.4)]`}
                    style={{
                      width: `${((values[p.id as keyof MetarValues] - p.min) / (p.max - p.min)) * 100}%`,
                    }}
                  />
                </div>
              </div>
            ))}
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
      <div className="flex-1 w-full border-l border-white/5 p-8 md:p-12 h-screen overflow-y-auto">
        {!encodedValues ? (
          <div className="h-full flex items-center justify-center text-slate-700 text-[10px] uppercase tracking-widest">
            {isLoading
              ? "Encoding Neural Representation..."
              : "Enter parameters and encode to begin"}
          </div>
        ) : (
          <div className="w-full flex flex-col gap-12">
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
