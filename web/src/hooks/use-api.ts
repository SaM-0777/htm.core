import { fetcher } from "@/utils/fetcher";
import useSWR from "swr";

const API_URL = process.env.NEXT_PUBLIC_API_URL!;

export function useEncode() {
  const mutationKey = `api/v1/encode`;
  const { data, error, isLoading } = useSWR([mutationKey], async ([url]) => {
    const res = await fetcher(`${API_URL}/${url}`);

    if (!res.success) {
      throw new Error(res.message || "Failed to fetch data");
    }

    return res;
  });

  return {
    data,
    error,
    isLoading,
  };
}
