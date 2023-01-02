package com.example.myapplication;

import android.Manifest;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.ParcelFileDescriptor;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {

    public static final int IMAGE_CAPTURE_CODE = 654;

    ImageView imageView;
    Button loadImage;
    Button detect;
    TextView resultText;

    Uri imageUri;
    Module module;
    Bitmap image = null;
    int rectHeight;
    int rectWidth;
    float resizeRatio;

    final int inputHeight = 600;
    final int inputWidth = 450;

    final String[] oneHotDecode = {"nv", "mel", "bkl", "bcc", "akiec", "df", "vasc"};

    private void openCamera() {
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.TITLE, "New Picture");
        values.put(MediaStore.Images.Media.DESCRIPTION, "From the Camera");
        imageUri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
        startActivityForResult(cameraIntent, IMAGE_CAPTURE_CODE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == IMAGE_CAPTURE_CODE && resultCode == RESULT_OK){
            image = uriToBitmap(imageUri);
            Bitmap tempBitmap = Bitmap.createBitmap(image.getWidth(), image.getHeight(), Bitmap.Config.RGB_565);
            Canvas canvas = new Canvas(tempBitmap);
            Rect dst = new Rect();
            dst.set(0, 0, image.getWidth(), image.getHeight());
            System.out.println("AJUNGEM AICI");
            canvas.drawBitmap(image, null, dst, null);
            int h = image.getHeight();
            int w = image.getWidth();
            resizeRatio = Math.min(h / (float) inputHeight, w / (float) inputWidth);
            rectHeight = (int)(inputHeight * resizeRatio);
            rectWidth = (int)(inputWidth * resizeRatio);
            int startX = (w / 2 - rectWidth / 2);
            int startY = (h / 2 - rectHeight / 2);
            Paint paint = new Paint();
            paint.setStyle(Paint.Style.STROKE);
            paint.setColor(Color.BLUE);
            paint.setStrokeWidth(10);
            canvas.drawRect(startX, startY, startX + rectWidth, startY + rectHeight, paint);
            imageView.setImageDrawable(new BitmapDrawable(getResources(), tempBitmap));
        }
    }

    public String assetFilePath(String assetName) {
        File file = new File(this.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = this.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        } catch (IOException e) {
            throw new RuntimeException();
        }
    }

    private Bitmap uriToBitmap(Uri selectedFileUri) {
        try {
            ParcelFileDescriptor parcelFileDescriptor =
                    getContentResolver().openFileDescriptor(selectedFileUri, "r");
            FileDescriptor fileDescriptor = parcelFileDescriptor.getFileDescriptor();
            Bitmap image = BitmapFactory.decodeFileDescriptor(fileDescriptor);

            parcelFileDescriptor.close();
            return image;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return  null;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = (ImageView)findViewById(R.id.image);
        loadImage = (Button)findViewById(R.id.button);
        detect = (Button)findViewById(R.id.detect);
        resultText = (TextView)findViewById(R.id.result_text);

        module = Module.load(assetFilePath("ham10k_optimized.pt"));

        loadImage.setOnClickListener(view -> {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED/* ||
                       checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_DENIED*/) {
                    String[] permission = {Manifest.permission.CAMERA};
                    requestPermissions(permission, 112);
                } else {
                    openCamera();
                }
            } else {
                openCamera();
            }
        });


        detect.setOnClickListener(view -> {
            if (image != null) {
                int h = image.getHeight();
                int w = image.getWidth();
                int startX = (w / 2 - rectWidth / 2);
                int startY = (h / 2 - rectHeight / 2);
                Bitmap centerPart = Bitmap.createBitmap(image, startX, startY, rectWidth, rectHeight);
                Matrix m = new Matrix();
                m.postScale(1.f / resizeRatio, 1.f / resizeRatio);
                Bitmap resized = Bitmap.createBitmap(centerPart, 0, 0, inputWidth, inputHeight, m, false);
                centerPart.recycle();
                Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resized,
                        TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
                System.out.println("Doing forward pass...");
                float[] out = module.forward(IValue.from(inputTensor)).toTensor().getDataAsFloatArray();
                int maxPos = 0;
                float maxVal = 0;
                for (int i = 0; i < out.length; i++) {
                    if (out[i] > maxVal) {
                        maxVal = out[i];
                        maxPos = i;
                    }
                }
                resultText.setText(oneHotDecode[maxPos]);
            }
        });
    }
}