# src/preprocess.py

# main() for the preprocessing pipeline

from pipeline import execute_pipeline as run


def main():
    cities_dict = {
        # "Utrecht": ["NL10636", "NL10641", 260],
        # "Amsterdam": ["NL49003", "NL49012", 240],  # nieuwe, diemenstraat
        "Rotterdam": ["NL01485", "NL01494", 344],  # hoogvliet, schiedam-a-arianstrat
    }

    # run the pipeline for the cities in the dictionary
    for city, locations in cities_dict.items():
        print(f"Running pipeline for {city}")

        run(
            ["O3", "NO2"],  # pass on contaminants;
            # for more variables, see pipeline/pipeline.py
            locations=locations,
            city_name=city,
            LOG=True,
            years=[2017, 2018, 2020, 2021, 2022, 2023],
            meteo_target=["temp", "dewP", "WD", "Wvh", "P", "SQ"],
        )


if __name__ == "__main__":
    main()
