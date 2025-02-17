# src/preprocess.py

# main() for the preprocessing pipeline

from pipeline import execute_pipeline as run


def main():
    cities_dict = {
        "Utrecht": ["NL10636", "NL10641", 260],
        # "Amsterdam": ["NL49014", "NL49012", ],
        # "Rotterdam": ["NL10418", "NL01493", 344],
    }

    # run the pipeline for the cities in the dictionary
    for city, locations in cities_dict.items():
        print(f"Running pipeline for {city}")

        run(
            ["PM25", "PM10", "O3", "NO2"],  # pass on contaminants;
            # for more variables, see pipeline/pipeline.py
            locations=locations,
            city_name=city,
            LOG=True,
        )

    # run(
    #     ["PM25", "PM10", "O3", "NO2"],  # pass on contaminants;
    #     # for more variables, see pipeline/pipeline.py
    # )


if __name__ == "__main__":
    main()
