# Lane-detection

In this project we detect car lanes using various image processing teechniques.

Process script should be run as follows:

process image|video input_path output_path [1|0]

The first argument should be either "image" or "video".
The second argument is the input path.
The second argument is the output path.
There is an optional fourth argument, if the fourth argument is "1", then all the steps in the pipeline will be shown.

## Examples:

process video ./project_video.mp4 output.mp4 1 (will be in debug mode)

process image ./test.jpg output.jpg
