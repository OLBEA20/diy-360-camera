from dataclasses import dataclass

from matplotlib import pyplot as plt


@dataclass
class ParameterResult:
    score: float
    del_x: int
    del_y: int
    aperture: int
    rotation_y: float


if __name__ == "__main__":
    results = []

    with open("./parameters_result.txt") as file:
        lines = file.readlines()

        for line in lines:
            clean_line = (
                line.replace("(", "")
                .replace(")", "")
                .replace(",", "")
                .replace("\n", "")
            )
            raw_parameters = clean_line.split(" ")
            results.append(
                ParameterResult(
                    score=float(raw_parameters[0]),
                    del_x=int(raw_parameters[1]),
                    del_y=int(raw_parameters[2]),
                    aperture=195 + int(raw_parameters[3]),
                    rotation_y=float(raw_parameters[4]),
                )
            )
    for delX in range(-20, 20):
        for delY in range(-10, 10):
            for rotation_y in range(-20, 20):
                results_to_graph = []
                for result in results:
                    if (
                        result.del_x == delX
                        and result.del_y == delY
                        and result.rotation_y == rotation_y / 4
                    ):
                        results_to_graph.append(result)

                xs = []
                ys = []
                for result in sorted(results_to_graph, key=lambda x: x.aperture):
                    xs.append(result.aperture)
                    ys.append(result.score)

                plt.plot(xs, ys)
                plt.savefig(f"./data/graphs/{delX}_{delY}_{rotation_y}.png")
                plt.close()
