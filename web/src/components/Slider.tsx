export interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit: string;
  onChange: (value: number) => void;
  disabled?: boolean;
}

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  unit,
  onChange,
  disabled = false,
}: SliderProps) {
  return (
    <div
      className={`group ${disabled ? "opacity-30 pointer-events-none" : "opacity-100"} transition-opacity duration-300 w-full`}
    >
      <div className="flex justify-between items-baseline mb-4">
        <label className="text-[10px] uppercase tracking-widest text-slate-500 font-medium group-hover:text-slate-300 transition-colors">
          {label}
        </label>
        <span className="text-sm font-mono text-white/90">
          {value}{" "}
          <span className="text-slate-500 text-[10px] ml-1">{unit}</span>
        </span>
      </div>
      <div className="relative h-0.5 bg-white/10 rounded-full flex items-center">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="absolute w-full appearance-none bg-transparent outline-none z-20 h-4 cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:shadow-[0_0_10px_rgba(255,255,255,0.8)] transition-all"
        />
        <div
          className="absolute h-full rounded-full bg-white shadow-[0_0_8px_rgba(255,255,255,0.4)]"
          style={{
            width: `${((value - min) / (max - min)) * 100}%`,
          }}
        />
      </div>
    </div>
  );
}
