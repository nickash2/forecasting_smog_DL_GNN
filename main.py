# from src.run_forecast import main
from src.graph_modelling.utils.rescale import rescale

if __name__ == "__main__":
    sensors = [
        ["NL01485", "NL01494"],  # Rotterdam
        ["NL10636", "NL10641"],  # Utrecht
        ["NL49003", "NL49012"],  # Amsterdam
    ]

    years = [2017, 2018, 2020, 2021, 2022, 2023]
    cities = ["rotterdam", "utrecht", "amsterdam"]

    rescale(sensors, years, cities)
