# Emotioner

Emotioner is an Android application designed to analyze images and predict the emotions depicted in them using a machine learning model. The app allows users to either capture images with the camera or upload images from the gallery. Once an image is selected, it is analyzed to determine the predominant emotion displayed.

## Features

- **Image Capture**: Take a photo using the device's camera.
- **Image Upload**: Select a photo from the device's gallery.
- **Emotion Analysis**: Analyze the selected image to predict the displayed emotion using a TensorFlow Lite model.

## Screenshots

![Main Screen](https://github.com/barlakshan/Face-Emotion-Detector-Mobile-Application-JAVA-TensorflowLite/assets/106991265/8e998829-ccbe-4a10-aa68-75e67842236c)
![Analysis Screen](https://github.com/barlakshan/Face-Emotion-Detector-Mobile-Application-JAVA-TensorflowLite/assets/106991265/ab734366-198f-4be6-96e2-bef2dfa843b8)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/barlakshan/MyApplication.git
    ```

2. Open the project in Android Studio.

3. Build and run the project on an Android device or emulator.

## Usage

1. Launch the app on your Android device.
2. Choose an option:
   - Tap on the **Camera** button to take a photo.
   - Tap on the **Upload** button to select an image from the gallery.
3. Once an image is selected, tap on the **Analyze** button to predict the emotion.
4. The predicted emotion will be displayed on the screen.

## Code Overview

### Main Components

- `SecondPage.java`: The main activity that handles image capture, image selection, and emotion analysis.
- `activity_second_page.xml`: The layout file for the `SecondPage` activity.

### Key Functions

- **Image Capture**:
    ```java
    camera_btn.setOnClickListener(new View.OnClickListener() {
        public void onClick(View v) {
            Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            if (intent.resolveActivity(getPackageManager()) != null) {
                startActivityForResult(intent, REQUEST_IMAGE_CAPTURE);
            }
        }
    });
    ```

- **Image Upload**:
    ```java
    upload_btn.setOnClickListener(new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(intent, REQUEST_IMAGE_PICK);
        }
    });
    ```

- **Emotion Analysis**:
    ```java
    analyse_btn.setOnClickListener(new View.OnClickListener() {
        @Override
        public void onClick(View v) {
            if (img != null) {
                try {
                    ModelUnquant model = ModelUnquant.newInstance(getApplicationContext());

                    Bitmap resizedImage = Bitmap.createScaledBitmap(img, 224, 224, true);
                    ByteBuffer byteBuffer = convertBitmapToByteBuffer(resizedImage);

                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
                    inputFeature0.loadBuffer(byteBuffer);

                    ModelUnquant.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    model.close();

                    String[] emotions = {"angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"};
                    float[] probabilities = outputFeature0.getFloatArray();

                    int maxIndex = 0;
                    float maxProbability = probabilities[0];
                    for (int i = 1; i < probabilities.length; i++) {
                        if (probabilities[i] > maxProbability) {
                            maxProbability = probabilities[i];
                            maxIndex = i;
                        }
                    }

                    tv.setText("Emotion: " + emotions[maxIndex]);

                } catch (IOException e) {
                    e.printStackTrace();
                    tv.setText("Error: " + e.getMessage());
                }
            } else {
                tv.setText("Please upload an image first.");
            }
        }
    });
    ```

### Helper Methods

- **Convert Bitmap to ByteBuffer**:
    ```java
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[224 * 224];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;

        for (int i = 0; i < 224; ++i) {
            for (int j = 0; j < 224; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
            }
        }
        return byteBuffer;
    }
    ```

## Model Details

The application uses a pre-trained TensorFlow Lite model (`ModelUnquant`) to analyze emotions. Ensure the model file is included in the `assets` folder of your project.

## Contributors

This project is a group assignment completed by the following contributors:

- **Contributor 1** - [Eranda-Uditha](https://github.com/Eranda-Uditha)
- **Contributor 2** - [SSandaruwanSrimal](https://github.com/SSandaruwanSrimal)
- **Contributor 3** - [Eranga0619](https://github.com/Eranga0619)


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- TensorFlow Lite for providing the framework for running machine learning models on mobile devices.
- Android development community for various tutorials and support.

