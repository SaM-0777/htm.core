from ecmwf.opendata import Client

client = Client()

client.retrieve(
    date="2026-05-22",  # or -1 for latest
    time="00",  # 00/06/12/18
    step=[0, 6, 12, 24, 48],
    type="fc",  # forecast
    param=["2t", "t2m"],  # 2 metre temperature
    target="ecmwf_data.grib2",
)
