package com.example.csc2231imggen;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;


import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.navigation.fragment.NavHostFragment;

import com.example.csc2231imggen.databinding.FragmentSecondBinding;
import com.example.csc2231imggen.ml.TfliteV2F32;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Random;


public class SecondFragment extends Fragment {

    private FragmentSecondBinding binding;
    //private static final String BASE_URL = "https://mobilegen-20e2753512ee.herokuapp.com/generate_image";
    private static final String BASE_URL = "https://4a00-35-240-186-244.ngrok-free.app/generate_image";
    private static final String TAG = "MainActivity Request Connection";

    private static final String[] cifar_labels = new String[]{"airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"};

    private Bitmap bitmap;

    private String user_prompt;
    private Integer version_type;

    private void generateImage(Integer class_num){
        try {
            TfliteV2F32 model = TfliteV2F32.newInstance(getActivity().getApplicationContext());

            // Creating inputs
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(64 * 64 * 3 * 4);
            byteBuffer.order(ByteOrder.nativeOrder());

            // Filling the ByteBuffer with random values (equivalent to torch.randn)
            Random random = new Random();
            while (byteBuffer.hasRemaining()) {
                byteBuffer.putFloat(random.nextFloat());
            }
            byteBuffer.rewind();

            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1}, DataType.FLOAT32);
            ByteBuffer b1 = ByteBuffer.allocateDirect(4);
            b1.putFloat(1);
            b1.rewind();
            inputFeature0.loadBuffer(b1); // torch.ones(1) equivalent
            TensorBuffer inputFeature1 = TensorBuffer.createFixedSize(new int[]{1}, DataType.FLOAT32);
            ByteBuffer b2 = ByteBuffer.allocateDirect(4);
            b2.putFloat(class_num);
            b2.rewind();
            inputFeature1.loadBuffer(b2); // setting value 6 equivalent

            TensorBuffer inputFeature2 = TensorBuffer.createFixedSize(new int[]{1, 3, 64, 64}, DataType.FLOAT32);
            inputFeature2.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            TfliteV2F32.Outputs outputs = model.process(inputFeature0, inputFeature1, inputFeature2);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            
            int[] shape = outputFeature0.getShape();

            Log.d("Shape of output is ", Arrays.toString(shape));

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            Log.e(TAG, "Exception occurred at model: " + e.getMessage());
        }
    }

    private void getImage(View view,String userPrompt,Integer version_type){
        // Sends a post request with x-www-form-urlencoded body containing "text" = userPrompt to the BASE_URL
        // The Response from the FAST API is an FileResponse containing an image.
        // The image received in the response must be stored into the bitmap.
//        Integer index = Arrays.binarySearch(cifar_labels, userPrompt);
//        generateImage(index);

        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    // Create a URL object with the base URL
                    URL url = new URL(BASE_URL);

                    // Open connection
                    HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                    connection.setRequestMethod("POST");
                    connection.setRequestProperty("Content-Type", "application/x-www-form-urlencoded");
                    connection.setDoOutput(true);

                    // Create the body of the request
                    //String body = "text=" + URLEncoder.encode(userPrompt, "UTF-8")+"&"+"version=" + URLEncoder.encode(version_type, "UTF-8");
                    String body = "text=" + URLEncoder.encode(userPrompt, "UTF-8") + "&" + "version=" + version_type;

                    // Write data to the connection output stream
                    OutputStream os = connection.getOutputStream();
                    os.write(body.getBytes());
                    os.flush();
                    os.close();

                    // Get response code
                    int responseCode = connection.getResponseCode();
                    if (responseCode == HttpURLConnection.HTTP_OK) {
                        // Read the response
                        InputStream inputStream = connection.getInputStream();
                        // Decode the response into Bitmap
                        bitmap = BitmapFactory.decodeStream(inputStream);
                        inputStream.close();

                        // Update ImageView with the new bitmap on UI thread
                        getActivity().runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                ImageView image_holder = view.findViewById(R.id.imageView);
                                image_holder.setImageBitmap(bitmap);
                            }
                        });
                    } else {
                        Log.e(TAG, "Failed to get image. Response code: " + responseCode);
                    }

                    // Disconnect the connection
                    connection.disconnect();
                } catch (Exception e) {
                    Log.e(TAG, "Exception occurred: " + e.getMessage());
                }
            }
        });
        thread.start();

    }

    @Override
    public View onCreateView(
            @NonNull LayoutInflater inflater, ViewGroup container,
            Bundle savedInstanceState
    ) {

        binding = FragmentSecondBinding.inflate(inflater, container, false);


        user_prompt = SecondFragmentArgs.fromBundle(getArguments()).getUserPromt();
        String[] parts = user_prompt.split("---");
        user_prompt = parts[0];
        version_type = Integer.parseInt(parts[1]);
        return binding.getRoot();

    }

    public void onViewCreated(@NonNull View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        String final_text = getString(R.string.result_text, user_prompt);

        TextView headerView = view.getRootView().findViewById(R.id.result);
        headerView.setText(final_text);
        getImage(view,user_prompt,version_type);


        binding.buttonSecond.setOnClickListener(v ->
            NavHostFragment.findNavController(SecondFragment.this)
                    .navigate(R.id.action_SecondFragment_to_FirstFragment)
         );

    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }

}