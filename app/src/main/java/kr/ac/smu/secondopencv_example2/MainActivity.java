package kr.ac.smu.secondopencv_example2;

import androidx.annotation.RawRes;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import android.annotation.TargetApi;
import android.content.Context;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.util.Arrays;
import java.util.List;

import static android.Manifest.permission.CAMERA;
import static java.lang.Double.NEGATIVE_INFINITY;
import static java.lang.Double.POSITIVE_INFINITY;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "opencv";
    private Mat matInput_raw1, matInput_raw2;
    private Mat matResult_pro1, matResult_pro2;

    private CameraBridgeViewBase mOpenCvCameraView_raw1;
    private CameraBridgeViewBase mOpenCvCameraView_raw2;
    private SurfaceView mOpenCvCameraView_pro1;
    private SurfaceView mOpenCvCameraView_pro2;

    /*private float mtx_array_fs[][];
    private float dist_array_fs[];*/

    private boolean bRaw1Prepared = false, bExported = false;
    private Mat matMtx1, matDist1, matMtx2, matDist2, matQ;

    public native void ConvertRGBtoGray(long matAddrInput, long matAddrResult);
    public native void GetRightImage(long matAddrInput, long matAddrReference, long matAddrMtx, long matAddrDist, long matAddrQ, long matAddrResult, long matAddrPoints, long matAddrColors);
    public native void GetLeftImage(long matAddrInput, long matAddrMtx, long matAddrDist, long matAddrResult);

    static {
        System.loadLibrary("native-lib");

        if (!OpenCVLoader.initDebug())
            Log.e("OpenCv", "Unable to load OpenCV");
        else
            Log.d("OpenCv", "OpenCV loaded");
    }

    public static String readTextFile(Context context, @RawRes int id){
        InputStream inputStream = context.getResources().openRawResource(id);
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();

        byte buffer[] = new byte[1024];
        int size;

        try {
            while ((size = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, size);
            }
            outputStream.close();
            inputStream.close();
        } catch (IOException e) {

        }

        return outputStream.toString();
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    mOpenCvCameraView_raw1.enableView();
                    mOpenCvCameraView_raw2.enableView();
                    /*mOpenCvCameraView_pro1.enableView();
                    mOpenCvCameraView_pro2.enableView();*/
                }
                break;
                default:
                {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        if (!OpenCVLoader.initDebug())
            Log.e("OpenCv", "Unable to load OpenCV");
        else
            Log.d("OpenCv", "OpenCV loaded");

        mOpenCvCameraView_raw1 = (CameraBridgeViewBase)findViewById(R.id.activity_surface_view_raw1);
        mOpenCvCameraView_raw1.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView_raw1.setCvCameraViewListener(new CvCameraViewListener2_raw1());
        mOpenCvCameraView_raw1.setCameraIndex(0); // front-camera(1),  back-camera(0)

        mOpenCvCameraView_raw2 = (CameraBridgeViewBase)findViewById(R.id.activity_surface_view_raw2);
        mOpenCvCameraView_raw2.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView_raw2.setCvCameraViewListener(new CvCameraViewListener2_raw2());
        mOpenCvCameraView_raw2.setCameraIndex(2); // front-camera(1),  back-camera(0)

        mOpenCvCameraView_pro1 = (SurfaceView)findViewById(R.id.activity_surface_view_pro1);
        mOpenCvCameraView_pro1.setVisibility(SurfaceView.VISIBLE);

        mOpenCvCameraView_pro2 = (SurfaceView)findViewById(R.id.activity_surface_view_pro2);
        mOpenCvCameraView_pro2.setVisibility(SurfaceView.VISIBLE);

        String strMtx1 = readTextFile(this, R.raw.mtxleft);

        String fsdata1[] = strMtx1.split("\\n");
        float i_mtx_array_fs1[][] = new float[fsdata1.length][];

        for (int i = 0; i < fsdata1.length; i++){
            String fdata[] = fsdata1[i].split(" ");
            i_mtx_array_fs1[i] = new float[fdata.length];

            for (int j = 0; j < fdata.length; j++) {
                i_mtx_array_fs1[i][j] = Float.parseFloat(fdata[j]);
                //Log.e("SecondOpenCV_Example2", new Integer(i).toString() + ", " + new Integer(j).toString() + ", " + new Float(mtx_array_fs[i][j]).toString());
            }
            //Log.e("SecondOpenCV_Example2", "\n");
        }

        String strDist1 = readTextFile(this, R.raw.distleft);

        String fdata1[] = strDist1.split(" ");
        float i_dist_array_fs1[] = new float[fdata1.length];

        for (int i = 0; i < fdata1.length; i++){
            i_dist_array_fs1[i] = Float.parseFloat(fdata1[i]);
            //Log.e("SecondOpenCV_Example2", new Integer(i).toString() + ", " + new Float(dist_array_fs[i]).toString());
        }

        matMtx1 = new Mat(3,3, CvType.CV_32F);
        for(int row=0;row<3;row++){
            for(int col=0;col<3;col++)
                matMtx1.put(row, col, i_mtx_array_fs1[row][col]);
        }

        matDist1 = new Mat(1,5,CvType.CV_32F);
        for(int col=0;col<5;col++)
            matDist1.put(0, col, i_dist_array_fs1[col]);

        String strMtx2 = readTextFile(this, R.raw.mtxright);

        String fsdata2[] = strMtx2.split("\\n");
        float i_mtx_array_fs2[][] = new float[fsdata2.length][];

        for (int i = 0; i < fsdata2.length; i++){
            String fdata[] = fsdata2[i].split(" ");
            i_mtx_array_fs2[i] = new float[fdata.length];

            for (int j = 0; j < fdata.length; j++) {
                i_mtx_array_fs2[i][j] = Float.parseFloat(fdata[j]);
                //Log.e("SecondOpenCV_Example2", new Integer(i).toString() + ", " + new Integer(j).toString() + ", " + new Float(mtx_array_fs[i][j]).toString());
            }
            //Log.e("SecondOpenCV_Example2", "\n");
        }

        String strDist2 = readTextFile(this, R.raw.distright);

        String fdata2[] = strDist2.split(" ");
        float i_dist_array_fs2[] = new float[fdata2.length];

        for (int i = 0; i < fdata2.length; i++){
            i_dist_array_fs2[i] = Float.parseFloat(fdata2[i]);
            //Log.e("SecondOpenCV_Example2", new Integer(i).toString() + ", " + new Float(dist_array_fs[i]).toString());
        }

        matMtx2 = new Mat(3,3, CvType.CV_32F);
        for(int row=0;row<3;row++){
            for(int col=0;col<3;col++)
                matMtx2.put(row, col, i_mtx_array_fs2[row][col]);
        }

        matDist2 = new Mat(1,5,CvType.CV_32F);
        for(int col=0;col<5;col++)
            matDist2.put(0, col, i_dist_array_fs2[col]);

        String strQ = readTextFile(this, R.raw.q);

        String fsdata3[] = strQ.split("\\n");
        float i_mtx_array_fs3[][] = new float[fsdata3.length][];

        for (int i = 0; i < fsdata3.length; i++){
            String fdata[] = fsdata3[i].split(" ");
            i_mtx_array_fs3[i] = new float[fdata.length];

            for (int j = 0; j < fdata.length; j++) {
                i_mtx_array_fs3[i][j] = Float.parseFloat(fdata[j]);
                //Log.e("SecondOpenCV_Example2", new Integer(i).toString() + ", " + new Integer(j).toString() + ", " + new Float(mtx_array_fs[i][j]).toString());
            }
            //Log.e("SecondOpenCV_Example2", "\n");
        }

        matQ = new Mat(4,4, CvType.CV_32F);
        for(int row=0;row<4;row++){
            for(int col=0;col<4;col++)
                matQ.put(row, col, i_mtx_array_fs3[row][col]);
        }
        /*Log.e("SecondOpenCV_Example2", "MTX");
        Log.e("SecondOpenCV_Example2", matMtx1.toString());

        for(int row=0;row<3;row++){
            for(int col=0;col<3;col++) {
                float[] element = new float[1];
                matMtx1.get(row, col, element);
                Log.e("SecondOpenCV_Example2", String.valueOf(element[0]));
            }
        }

        Log.e("SecondOpenCV_Example2", "DIST");
        Log.e("SecondOpenCV_Example2", matDist1.toString());

        for(int col=0;col<5;col++) {
            float[] element = new float[1];
            matDist1.get(0, col, element);
            Log.e("SecondOpenCV_Example2", String.valueOf(element[0]));
        }*/
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView_raw1 != null)
            mOpenCvCameraView_raw1.disableView();
        if (mOpenCvCameraView_raw2 != null)
            mOpenCvCameraView_raw2.disableView();
        /*if (mOpenCvCameraView_pro1 != null)
            mOpenCvCameraView_pro1.disableView();
        if (mOpenCvCameraView_pro2 != null)
            mOpenCvCameraView_pro2.disableView();*/
    }

    @Override
    public void onResume()
    {
        super.onResume();

        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "onResume :: Internal OpenCV library not found.");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "onResume :: OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();

        if (mOpenCvCameraView_raw1 != null)
            mOpenCvCameraView_raw1.disableView();

        if (mOpenCvCameraView_raw2 != null)
            mOpenCvCameraView_raw2.disableView();

        /*if (mOpenCvCameraView_pro1 != null)
            mOpenCvCameraView_pro1.disableView();

        if (mOpenCvCameraView_pro2 != null)
            mOpenCvCameraView_pro2.disableView();*/
    }

    private class CvCameraViewListener2_raw1 implements CameraBridgeViewBase.CvCameraViewListener2 {
        @Override
        public void onCameraViewStarted(int width, int height) {
            Log.e("raw1", "width: "+Integer.toString(width)+", height: "+Integer.toString(height));
        }

        @Override
        public void onCameraViewStopped() {

        }

        @Override
        public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
            matInput_raw1 = inputFrame.rgba();
            matResult_pro1 = new Mat();

            if (matInput_raw1 != null && matResult_pro1 != null && matMtx1 != null && matDist1 != null)
                GetLeftImage(matInput_raw1.getNativeObjAddr(), matMtx1.getNativeObjAddr(), matDist1.getNativeObjAddr(), matResult_pro1.getNativeObjAddr());
            else
                Log.e("SecondOpenCV_Example2", "Some data are null!!!" + matInput_raw1 + ", " + matResult_pro1 + ", " + matMtx1 + ", " + matDist1);

            bRaw1Prepared = true;

            return matResult_pro1;
        }
    }

    private class CvCameraViewListener2_raw2 implements CameraBridgeViewBase.CvCameraViewListener2 {
        @Override
        public void onCameraViewStarted(int width, int height) {
            Log.e("raw2", "width: "+Integer.toString(width)+", height: "+Integer.toString(height));
        }

        @Override
        public void onCameraViewStopped() {

        }

        void exportToFile(String strFilename, Mat matPoints, Mat matColors) {
            //filestream.open(filename, std::ios::out);
            String strHeader = readTextFile(getApplicationContext(), R.raw.plyheader), strContents = new String();
            FileOutputStream fosContents = null, fosHeader = null;

            try {
                //Log.d("secondopencv_example2", matPoints.dump());
                //Log.d("secondopencv_example2", matPoints.size().toString());
                int vert_num = 0;
                int rows = matPoints.rows(), cols = matPoints.cols();

                fosContents = getApplicationContext().openFileOutput(strFilename+"_con", Context.MODE_PRIVATE);

                for(int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        double[] dTemp1 = matPoints.get(i, j);
                        if (dTemp1[0] == POSITIVE_INFINITY || dTemp1[0] == NEGATIVE_INFINITY || dTemp1[1] == POSITIVE_INFINITY || dTemp1[1] == NEGATIVE_INFINITY || dTemp1[2] == POSITIVE_INFINITY || dTemp1[2] == NEGATIVE_INFINITY) continue;

                        double[] dTemp2= matColors.get(i, j);
                        strContents += dTemp1[0] + " " + dTemp1[1] + " " + dTemp1[2] + " " + (byte)dTemp2[2] + " " + (byte)dTemp2[1] + " " + (byte)dTemp2[0] + "\r\n";

                        fosContents.write(strContents.getBytes());
                        strContents = "";

                        vert_num++;
                    }
                }

                fosContents.flush();
                fosContents.close();

                fosHeader = getApplicationContext().openFileOutput(strFilename, Context.MODE_PRIVATE);

                //long numElem = matPoints.total();
                strHeader = strHeader.replace("vert_num", new Long(vert_num).toString());
                fosHeader.write(strHeader.getBytes());
                //fosHeader.write(strContents.getBytes());
                //strContents = "";

                //FileInputStream fis = getApplicationContext().openFileInput(strFilename+"_temp");
                //int c;
                //while ((c = fis.read()) != -1) {
                //    fos.write(c);
                //}
                //fis.close();

                fosHeader.flush();
                fosHeader.close();
            } catch (Exception e) {
                Log.e("", e.getMessage());
            } finally {
                if (fosContents != null) {
                    fosContents = null;
                }
                if (fosHeader != null) {
                    fosHeader = null;
                }
            }
            /*OutputStreamWriter osw = new OutputStreamWriter(context.openFileOutput(strFilename, Context.MODE_PRIVATE));

            // MARK: Header writing
            filestream << "ply" << std::endl <<
                "format " << enum2str[format] << " 1.0" << std::endl <<
                "comment file created using code by Cedric Menard" << std::endl <<
                "element vertex " << numElem << std::endl <<
                "property float x" << std::endl <<
                "property float y" << std::endl <<
                "property float z" << std::endl <<
                "property uchar red" << std::endl <<
                "property uchar green" << std::endl <<
                "property uchar blue" << std::endl <<
                "end_header" << std::endl;

            // MARK: Data writing
            // Pointer to data
            const float* pData = data.ptr<float>(0);
            const unsigned char* pColor = colors.ptr<unsigned char>(0);
            const unsigned long numIter = 3*numElem;                            // Number of iteration (3 channels * numElem)
            const bool hostIsLittleEndian = isLittleEndian();

            float_t bufferXYZ;                                                 // Coordinate buffer for float type

            for (unsigned long i = 0; i<numIter; i+=3) {                            // Loop through all elements
                for (unsigned int j = 0; j<3; j++) {                                // Loop through 3 coordinates
                    filestream << std::setprecision(9) << pData[i+j] << " ";
                }
                for (int j = 2; j>=0; j--) {
                    // OpenCV uses BGR format, so the order of writing is reverse to comply with the RGB format
                    filestream << (unsigned short)pColor[i+j] << (j==0?"":" ");                     // Loop through RGB
                }
                filestream << std::endl;                                            // End if element line
            }*/
        }

        @Override
        public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
            matInput_raw2 = inputFrame.rgba();
            matResult_pro2 = new Mat(480, 640, CvType.CV_8UC4);

            //matResult_pro2 = new Mat(352, 288, CvType.CV_8UC4);
            //Mat matTemp  = new Mat(480, 640, CvType.CV_32FC4);
            Mat matPoints = new Mat();
            Mat matColors = new Mat();

            while(!bRaw1Prepared);

            if (matInput_raw2 != null && matResult_pro2 != null && matMtx2 != null && matDist2 != null) {
                long sTime = System.nanoTime();
                GetRightImage(matInput_raw2.getNativeObjAddr(), matResult_pro1.getNativeObjAddr(), matMtx2.getNativeObjAddr(), matDist2.getNativeObjAddr(), matQ.getNativeObjAddr(), matResult_pro2.getNativeObjAddr(), matPoints.getNativeObjAddr(), matColors.getNativeObjAddr());
                long eTime = System.nanoTime() - sTime;
                if (!bExported) {
                    exportToFile("out.ply", matPoints, matColors);
                    bExported = true;
                }
                Log.d("Elapsed time (Total)",  Long. toString(eTime) + " nsec, " + Float.toString(1000000000f/ eTime) + " FPS");
            }
            else
                Log.e("SecondOpenCV_Example2", "Some data are null!!!" + matInput_raw2 + ", " + matResult_pro2 + ", " + matMtx2 + ", " + matDist2);

            //matResult_pro2.convertTo(matTemp, CvType.CV_16FC3, 1.0/255.0);
            //Imgproc.cvtColor(matResult_pro2, matTemp, Imgproc.COLOR_RGB2BGR);
            matResult_pro2.convertTo(matResult_pro2, CvType.CV_8UC4);

            bRaw1Prepared = false;

            //String strDir = getFilesDir().getAbsolutePath();
            //Imgcodecs.imwrite(strDir + "/test.png", matTemp);
            //Imgcodecs.imwrite(strDir + "/test.png", matResult_pro2);

            //return matTemp;
            return matResult_pro2;
        }
    }

    private class CvCameraViewListener2_pro1 extends SurfaceView {
        public CvCameraViewListener2_pro1(Context context) {
            super(context);
        }
    }

    private class CvCameraViewListener2_pro2 extends SurfaceView {
        public CvCameraViewListener2_pro2(Context context) {
            super(context);
        }
    }

    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        //return Collections.singletonList(mOpenCvCameraView);
        return Arrays.asList(mOpenCvCameraView_raw1, mOpenCvCameraView_raw2); //, mOpenCvCameraView_pro1, mOpenCvCameraView_pro2);
    }

    //여기서부턴 퍼미션 관련 메소드
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 200;

    protected void onCameraPermissionGranted() {
        List<? extends CameraBridgeViewBase> cameraViews = getCameraViewList();
        if (cameraViews == null) {
            return;
        }
        for (CameraBridgeViewBase cameraBridgeViewBase: cameraViews) {
            if (cameraBridgeViewBase != null) {
                cameraBridgeViewBase.setCameraPermissionGranted();
            }
        }
    }

    @Override
    protected void onStart() {
        super.onStart();
        boolean havePermission = true;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(CAMERA) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
                havePermission = false;
            }
        }
        if (havePermission) {
            onCameraPermissionGranted();
        }
    }

    @Override
    @TargetApi(Build.VERSION_CODES.M)
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            onCameraPermissionGranted();
        }else{
            showDialogForPermission("앱을 실행하려면 퍼미션을 허가하셔야합니다.");
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }

    @TargetApi(Build.VERSION_CODES.M)
    private void showDialogForPermission(String msg) {
        AlertDialog.Builder builder = new AlertDialog.Builder( MainActivity.this);
        builder.setTitle("알림");
        builder.setMessage(msg);
        builder.setCancelable(false);
        builder.setPositiveButton("예", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface dialog, int id){
                requestPermissions(new String[]{CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
            }
        });
        builder.setNegativeButton("아니오", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface arg0, int arg1) {
                finish();
            }
        });
        builder.create().show();
    }
}