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

import org.pytorch.Device;
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity implements ImageReader.OnImageAvailableListener {

    public static final int IMAGE_CAPTURE_CODE = 654;

    FrameLayout imageFrameLayout;
    ImageView imageView;
    Button loadImage;
    Button detect;
    TextView resultText;

    Uri imageUri;
    Module module_feature;
    Module module_classify;
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

    public Tensor getCurImageTensor() {
        int w = curImage.getWidth();
        int h = curImage.getHeight();
        int startX = (w / 2 - rectWidth / 2);
        int startY = (h / 2 - rectHeight / 2);
        Bitmap centerPart = Bitmap.createBitmap(curImage, startX, startY, rectWidth, rectHeight);
        Matrix m = new Matrix();
        m.postScale(1.f / resizeRatio, 1.f / resizeRatio);
        Bitmap resized = Bitmap.createBitmap(centerPart, 0, 0, rectWidth, rectHeight, m, false);
        System.out.println("SIZE OF BITMAP IS: " + resized.getHeight() + " " + resized.getWidth());
        centerPart.recycle();
        return TensorImageUtils.bitmapToFloat32Tensor(resized,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
    }

    public float[][] readMatrix(String path) {
        try (BufferedReader fileReader = new BufferedReader(new FileReader(path))) {
            int n = Integer.parseInt(fileReader.readLine());
            int m = Integer.parseInt(fileReader.readLine());
            float[][] data = new float[n][m];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++)
                    data[i][j] = Float.parseFloat(fileReader.readLine());
            }
            return data;
        } catch (IOException e) {
            return null;
        }
    }

    public float[][] matrixMul(float[][] m1, float[][] m2) {
        if (m1[0].length != m2.length)
            throw new RuntimeException();

        float[][] res = new float[m1.length][m2[0].length];
        for (int i = 0; i < m1.length; i++) {
            for (int j = 0; j < m2[0].length; j++)
                res[i][j] = 0.f;
        }

        for (int k = 0; k < m1[0].length; k++) {
            for (int i = 0; i < m1.length; i++) {
                for (int j = 0; j < m2[0].length; j++)
                    res[i][j] += m1[i][k] * m2[k][j];
            }
        }
        return res;
    }

    public float[][] matrixDif(float[][] m1, float[][] m2) {
        if (m1.length != m2.length || m1[0].length != m2[0].length)
            throw new RuntimeException();

        float[][] res = new float[m1.length][m1[0].length];
        for (int i = 0; i < m1.length; i++) {
            for (int j = 0; j < m1[0].length; j++)
                res[i][j] = m1[i][j] - m2[i][j];
        }
        return res;
    }

    public float[][] matrixTranspose(float[][] m) {
        float[][] res = new float[m[0].length][m.length];

        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[0].length; j++)
                res[j][i] = m[i][j];
        }
        return res;
    }
    public float mahalanobis_distance(float[][] feature_map, float[][] mean_feature_map, float[][] covar_inverse) {
        float[][] adjusted_feature_map = matrixDif(feature_map, mean_feature_map);
        return matrixMul(matrixMul(adjusted_feature_map, covar_inverse), matrixTranspose(adjusted_feature_map))[0][0];
    }

    public float[][] vecToMatrix(float[] v) {
        float[][] res = new float[1][v.length];
        System.arraycopy(v, 0, res[0], 0, v.length);
        return res;
    }

    public float rmd_confidence(float[][] feature_map, float[][] mean_feature_maps, float[][] mean_feature_map_0, float[][] covar_inverse, float[][] covar_0_inverse) {
        float md_0 = mahalanobis_distance(feature_map, mean_feature_map_0, covar_0_inverse);

        ArrayList<Float> rmd = new ArrayList<>();
        for (float[] mean_feature_map : mean_feature_maps)
            rmd.add(mahalanobis_distance(feature_map, vecToMatrix(mean_feature_map), covar_inverse) - md_0);

        float min_val = rmd.get(0);
        for (int i = 1; i < rmd.size(); i++)
            min_val = rmd.get(i) < min_val ? rmd.get(i) : min_val;
        return -min_val;
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
        module_feature = Module.load(assetFilePath("ham10k_feature_mobile.pt"));
        module_classify = Module.load(assetFilePath("ham10k_classify_mobile.pt"));
        float[][] mean_feature_map_0 = readMatrix(assetFilePath("mean_feature_map_0.matrix"));
        float[][] mean_feature_maps = readMatrix(assetFilePath("mean_feature_maps.matrix"));
        float[][] covar_0_inverse = readMatrix(assetFilePath("covar_0_inverse.matrix"));
        float[][] covar_inverse = readMatrix(assetFilePath("covar_inverse.matrix"));

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
                Thread t = new Thread(() -> {
                    Tensor inputTensor = getCurImageTensor();
                    Tensor feature = module_feature.forward(IValue.from(inputTensor)).toTensor();
                    float confidence = rmd_confidence(vecToMatrix(feature.getDataAsFloatArray()), mean_feature_maps, mean_feature_map_0, covar_inverse, covar_0_inverse);

                    if (imageCaptured) {
                        System.out.println("IND confidence score: " + String.format("%.02f", confidence));
                        if (confidence < 0.f)
                            resultText.setText("Image doesn't seem to depict a lesion");
                        else
                            resultText.setText("Image looks good");
                    }
                });
                t.start();
            } else {
                loadImage.setText("Capture Image");
            }
            imageCaptured = !imageCaptured;
        });

        detect.setOnClickListener(view -> {
            if (curImage != null && imageCaptured) {
                Thread t = new Thread(() -> {
                    Tensor inputTensor = getCurImageTensor();
                    runOnUiThread(() -> {
                        resultText.setText("Analyzing...");
                    });
                    Tensor feature = module_feature.forward(IValue.from(inputTensor)).toTensor();
                    System.out.println(Arrays.toString(feature.shape()));
                    float[] out = module_classify.forward(IValue.from(feature)).toTensor().getDataAsFloatArray();
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