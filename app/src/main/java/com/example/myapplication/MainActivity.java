package com.example.myapplication;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.drawable.BitmapDrawable;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraManager;
import android.media.Image;
import android.media.ImageReader;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.util.Size;
import android.view.Surface;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import android.app.Fragment;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity implements ImageReader.OnImageAvailableListener {

    public static final int IMAGE_CAPTURE_CODE = 654;

    FrameLayout imageFrameLayout;
    ImageView imageView;
    Button loadImage;
    Button detect;
    TextView resultText;

    Uri imageUri;
    Module module;
    Bitmap curImage = null;
    int rectHeight;
    int rectWidth;
    float resizeRatio;

    final int inputHeight = 600;
    final int inputWidth = 450;

    final String[] oneHotDecode = {
            "melanocytic nevi",
            "melanoma",
            "benign keratosis-like lesion",
            "basal cell carcinoma",
            "bowen's disease",
            "dermatofibroma",
            "vascular lesion"};

    int previewHeight;
    int previewWidth;
    int[] rgbBytes;
    int sensorOrientation;

    boolean imageCaptured = false;

    private void openCamera() {
        CameraManager manager = (CameraManager)getSystemService(Context.CAMERA_SERVICE);
        String cameraId = null;
        try {
            cameraId = manager.getCameraIdList()[0];
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
        Fragment fragment;
        CameraConnectionFragment camera2Fragment = CameraConnectionFragment.Companion.newInstance(
                (CameraConnectionFragment.ConnectionCallback) (size, cameraRotation) -> {
                    previewHeight = size.getHeight();
                    previewWidth = size.getWidth();
                    sensorOrientation = cameraRotation - getScreenOrientation();
                },
                this,
                R.layout.fragment_camera_connection,
                new Size(600, 600)
        );
        camera2Fragment.setCamera(cameraId);
        fragment = camera2Fragment;
        getFragmentManager().beginTransaction().replace(R.id.frame, fragment).commit();
    }

    protected Integer getScreenOrientation() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            case Surface.ROTATION_0:
                return 0;
        }
        return 0;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == IMAGE_CAPTURE_CODE && grantResults.length > 0){
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                resultText.setText("");
                openCamera();
            } else {
                resultText.setText("PLEASE GRANT CAMERA PERMISSIONS BY TOUCHING AREA ABOVE");
            }
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

    private void requestAppPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
                String[] permission = {Manifest.permission.CAMERA};
                requestPermissions(permission, IMAGE_CAPTURE_CODE);
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageFrameLayout = (FrameLayout)findViewById(R.id.frame);
        imageView = (ImageView)findViewById(R.id.image);
        loadImage = (Button)findViewById(R.id.button);
        detect = (Button)findViewById(R.id.detect);
        resultText = (TextView)findViewById(R.id.result_text);

        module = Module.load(assetFilePath("ham10k_optimized.pt"));

        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M || checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
            openCamera();
        else
            requestAppPermissions();

        imageView.setOnClickListener(view -> {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
                requestAppPermissions();
            }
        });

        loadImage.setOnClickListener(view -> {
            if (resultText.getText() == "Analyzing...")
                return;
            resultText.setText("");
            if (!imageCaptured) {
                loadImage.setText("Take Another Image");
            } else {
                loadImage.setText("Capture Image");
            }
            imageCaptured = !imageCaptured;
        });

        detect.setOnClickListener(view -> {
            if (curImage != null && imageCaptured) {
                Thread t = new Thread(() -> {int h = curImage.getHeight();
                    int w = curImage.getWidth();
                    int startX = (w / 2 - rectWidth / 2);
                    int startY = (h / 2 - rectHeight / 2);
                    Bitmap centerPart = Bitmap.createBitmap(curImage, startX, startY, rectWidth, rectHeight);
                    Matrix m = new Matrix();
                    m.postScale(1.f / resizeRatio, 1.f / resizeRatio);
                    Bitmap resized = Bitmap.createBitmap(centerPart, 0, 0, rectWidth, rectHeight, m, false);
                    centerPart.recycle();
                    Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resized,
                            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
                    runOnUiThread(() -> {
                        resultText.setText("Analyzing...");
                    });
                    float[] out = module.forward(IValue.from(inputTensor)).toTensor().getDataAsFloatArray();
                    System.out.println("Result vector is: " + Arrays.toString(out));
                    int maxPos = 0;
                    float maxVal = 0;
                    for (int i = 0; i < out.length; i++) {
                        if (out[i] > maxVal) {
                            maxVal = out[i];
                            maxPos = i;
                        }
                    }
                    int finalMaxPos = maxPos;
                    runOnUiThread(() -> {
                        resultText.setText(oneHotDecode[finalMaxPos]);
                    });
                });
                t.start();
            }
        });
    }

    private boolean isProcessingFrame = false;
    private byte[][] yuvBytes = {null, null, null};
    private int yRowStride = 0;
    private Runnable imageConverter;
    private Runnable postInferenceCallback;

    @Override
    public void onImageAvailable(ImageReader reader) {
        // We need wait until we have some size from onPreviewSizeChosen
        if (previewWidth == 0 || previewHeight == 0) {
            return;
        }
        if (rgbBytes == null) {
            rgbBytes = new int[previewWidth * previewHeight];
        }
        try {
            Image image = reader.acquireLatestImage();
            if (image == null)
                return;
            if (isProcessingFrame) {
                image.close();
                return;
            }
            isProcessingFrame = true;
            Image.Plane[] planes = image.getPlanes();
            fillBytes(planes, yuvBytes);
            yRowStride = planes[0].getRowStride();
            int uvRowStride = planes[1].getRowStride();
            int uvPixelStride = planes[1].getPixelStride();
            imageConverter = () -> {
                ImageUtils.INSTANCE.convertYUV420ToARGB8888(
                        yuvBytes[0],
                        yuvBytes[1],
                        yuvBytes[2],
                        previewWidth,
                        previewHeight,
                        yRowStride,
                        uvRowStride,
                        uvPixelStride,
                        rgbBytes
                );
            };
            postInferenceCallback = () -> {
                image.close();
                isProcessingFrame = false;
            };
            processImage();
        } catch (Exception e){
            e.printStackTrace();
            return;
        }
    }


    private void processImage() {
        imageConverter.run();
        Bitmap rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
        rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);
        if (imageView != null && !imageCaptured) {
            Matrix rotationMatrix = new Matrix();
            rotationMatrix.setRotate((float)sensorOrientation);
            curImage =  Bitmap.createBitmap(rgbFrameBitmap, 0, 0, rgbFrameBitmap.getWidth(), rgbFrameBitmap.getHeight(), rotationMatrix, true);

            Bitmap tempBitmap = Bitmap.createBitmap(curImage.getWidth(), curImage.getHeight(), Bitmap.Config.RGB_565);
            Canvas canvas = new Canvas(tempBitmap);
            Rect dst = new Rect();
            dst.set(0, 0, curImage.getWidth(), curImage.getHeight());
            canvas.drawBitmap(curImage, null, dst, null);
            int h = tempBitmap.getHeight();
            int w = tempBitmap.getWidth();
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

            runOnUiThread(() -> {
                imageView.setBackground(new BitmapDrawable(getResources(), tempBitmap));
            });
        }
        postInferenceCallback.run();
    }

    private static void fillBytes(final Image.Plane[] planes, final byte[][] yuvBytes) {
        for (int i = 0; i < planes.length; ++i) {
            final ByteBuffer buffer = planes[i].getBuffer();
            if (yuvBytes[i] == null || yuvBytes[i].length != buffer.capacity()) {
                yuvBytes[i] = new byte[buffer.capacity()];
            }
            buffer.get(yuvBytes[i]);
        }
    }
}