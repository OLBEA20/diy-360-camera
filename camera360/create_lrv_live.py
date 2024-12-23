import os
import subprocess
import sys
import time


if __name__ == "__main__":
    frame_path = sys.argv[1]

    processed_frames = []
    frames = os.listdir(frame_path)
    indexed_frames = [(int(frame.split(".")[0][4:]), frame) for frame in frames]
    ordered_frames = sorted(frames)
    frames_to_process = ordered_frames[len(processed_frames) :]

    input_video = "output.mpeg"
    output_video = "input.mpeg"

    temp_video = "temp.mpeg"

    while frames_to_process:
        input_video, output_video = output_video, input_video
        params = []
        for frame in frames_to_process:
            params.append(frame)

        with open("frames.txt", "w") as file:
            for frame in frames_to_process:
                file.write(f"file {frame_path}/{frame}")
                file.write("\n")

        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                "frames.txt",
                temp_video,
            ]
        )

        if os.path.exists(input_video):
            with open("videos.txt", "w") as file:
                file.write(f"file {input_video}\n")
                file.write(f"file {temp_video}")

                subprocess.run(
                    [
                        "ffmpeg",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        "videos.txt",
                        "-c",
                        "copy",
                        output_video,
                    ]
                )
        else:
            os.rename(temp_video, input_video)

        if os.path.exists(output_video):
            os.rename(output_video, input_video)

        processed_frames.extend(frames_to_process)

        time.sleep(2)

        frames = os.listdir(frame_path)
        indexed_frames = [(int(frame.split(".")[0][4:]), frame) for frame in frames]
        ordered_frames = sorted(frames)
        frames_to_process = ordered_frames[len(processed_frames) :]

    os.rename(input_video, "lrv.mpeg")
