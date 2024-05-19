package com.example.myapplication;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.example.myapplication.ml.ModelUnquant;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class SecondPage extends AppCompatActivity {

    private static final int REQUEST_IMAGE_CAPTURE = 100;
    private static final int REQUEST_IMAGE_PICK = 101;
    private ImageView imageView;
    private Button upload_btn, analyse_btn, camera_btn;
    private TextView tv;
    private Bitmap img;

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_second_page);

        imageView = findViewById(R.id.imageView);
        upload_btn = findViewById(R.id.upload_btn);
        camera_btn = findViewById(R.id.camera_btn);
        analyse_btn = findViewById(R.id.analyse_btn);
        tv = findViewById(R.id.textView);

        camera_btn.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                if (intent.resolveActivity(getPackageManager()) != null) {
                    startActivityForResult(intent, REQUEST_IMAGE_CAPTURE);
                }
            }
        });

        upload_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(intent, REQUEST_IMAGE_PICK);
            }
        });

        analyse_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (img != null) {
                    try {
                        ModelUnquant model = ModelUnquant.newInstance(getApplicationContext());

                        Bitmap resizedImage = Bitmap.createScaledBitmap(img, 224, 224, true);

                        // Convert Bitmap to ByteBuffer
                        ByteBuffer byteBuffer = convertBitmapToByteBuffer(resizedImage);

                        // Creates inputs for reference.
                        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
                        inputFeature0.loadBuffer(byteBuffer);

                        // Runs model inference and gets result.
                        ModelUnquant.Outputs outputs = model.process(inputFeature0);
                        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                        // Releases model resources if no longer used.
                        model.close();

                        // Define emotion labels
                        String[] emotions = {"angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"};

                        // Get the output probabilities
                        float[] probabilities = outputFeature0.getFloatArray();

                        // Find the index of the highest probability
                        int maxIndex = 0;
                        float maxProbability = probabilities[0];
                        for (int i = 1; i < probabilities.length; i++) {
                            if (probabilities[i] > maxProbability) {
                                maxProbability = probabilities[i];
                                maxIndex = i;
                            }
                        }

                        // Display the corresponding emotion label
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
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && data != null) {
            if (requestCode == REQUEST_IMAGE_CAPTURE) {
                Bundle extras = data.getExtras();
                if (extras != null) {
                    img = (Bitmap) extras.get("data");
                    imageView.setImageBitmap(img);
                }
            } else if (requestCode == REQUEST_IMAGE_PICK) {
                Uri selectedImageUri = data.getData();
                if (selectedImageUri != null) {
                    try {
                        InputStream imageStream = getContentResolver().openInputStream(selectedImageUri);
                        img = BitmapFactory.decodeStream(imageStream);
                        imageView.setImageBitmap(img);
                    } catch (IOException e) {
                        e.printStackTrace();
                        tv.setText("Error: " + e.getMessage());
                    }
                }
            }
        }
    }

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
}
