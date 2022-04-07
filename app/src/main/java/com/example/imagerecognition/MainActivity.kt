package com.example.imagerecognition

import android.Manifest
import android.content.ActivityNotFoundException
import android.content.ContentValues
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import com.example.imagerecognition.ml.MobilenetV110224Quant
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File

class MainActivity : AppCompatActivity() {

    //declare the variables
    private lateinit var image: ImageView
    private lateinit var select: Button
    private lateinit var camera: Button
    private lateinit var predict: Button
    private lateinit var text: TextView
    private lateinit var bitmap: Bitmap
    private var imageUri: Uri? = null
    private var label: String = " "

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        //initialise the view variables
        image = findViewById(R.id.image)
        select = findViewById(R.id.selectButton)
        camera = findViewById(R.id.cameraButton)
        predict = findViewById(R.id.predictButton)
        text = findViewById(R.id.text)

        //maintain state when the configuration changes
        if (savedInstanceState != null) {
            //get and display the image
            imageUri = savedInstanceState.getParcelable("imageUri")
            image.setImageURI(imageUri)
            //get and display the label
            label = savedInstanceState.getString("label").toString()
            text.text = label
        }

        //request permission to use the camera
        val requestSinglePermissionLauncher =
            registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
                if (isGranted) {
                    Log.i(ContentValues.TAG, "Camera permission granted")
                }
            }
        requestSinglePermissionLauncher.launch(Manifest.permission.CAMERA)

        //get the labels from the textfile
        val fileName = "labels.txt"
        val input = application.assets.open(fileName).bufferedReader().use { it.readText() }
        val labelList = input.split("\n")

        //set up the contract to access images stored in the phone
        val pickImage = registerForActivityResult(ActivityResultContracts.GetContent()) {
            it?.let {
                imageUri = it
                //set image in the imageview
                image.setImageURI(imageUri)
            }
        }

        //set up the contract for taking a picture using the camera
        val takePicture =
            registerForActivityResult(ActivityResultContracts.TakePicture()) { success ->
                if (success) {
                    image.setImageURI(imageUri)
                    //convert to bitmap
                    val uri: Uri? = imageUri
                    bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
                }
            }

        //set onclicklistener to select an image from the phone when select button is pressed
        select.setOnClickListener {
            //clear the textview
            text.text = " "
            //clear the imageview
            image.setImageURI(null)
            //launch the actvity
            pickImage.launch("image/*")
        }

        //set onclicklistener to take a picture with the phone camera
        camera.setOnClickListener {
            //clear the textview
            text.text = " "
            //clear the imageview
            image.setImageURI(null)
            //try to take an image with the camera and get the uri
            try {
                imageUri = FileProvider.getUriForFile(
                    this,
                    BuildConfig.APPLICATION_ID + ".provider", createImageFile()
                )
                takePicture.launch(imageUri)
            }
            //catch exception incase camera cannot be accessed/is disabled
            catch (e: ActivityNotFoundException) {
                Log.e(ContentValues.TAG, "Camera not enabled on device")
                Toast.makeText(this, "Cannot access device camera", Toast.LENGTH_SHORT).show()
            }
        }

        //set the onclicklistener to predict what is in the selected image
        predict.setOnClickListener {
            //if there is no image in the imageview, display a message to the user
            if (image.drawable == null) {
                //clear the textview
                text.text = " "
                //display a message to the user
                Toast.makeText(this, "No image provided", Toast.LENGTH_SHORT).show()
            }
            //if there is an image, use the image to make a prediction
            else {
                //convert image to bitmap
                val uri: Uri? = imageUri
                bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)

                //resize the image to be fed into the model
                val resizedBitmap: Bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

                val model = MobilenetV110224Quant.newInstance(this)

                //creates inputs for reference.
                val inputFeature0 =
                    TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)

                //create bytebuffer from the resized bitmap
                val tensorBuffer = TensorImage.fromBitmap(resizedBitmap)
                val byteBuffer = tensorBuffer.buffer

                inputFeature0.loadBuffer(byteBuffer)

                //runs model inference and gets result.
                val outputs = model.process(inputFeature0)
                val outputFeature0 = outputs.outputFeature0AsTensorBuffer

                //get correct label for the prediction
                val max = getMax(outputFeature0.floatArray)

                label = labelList[max]

                //show the prediction in the textview
                text.text = label

                //releases model resources if no longer used.
                model.close()
            }
        }
    }

    //create a temporary image file if taking a picture with the camera
    private fun createImageFile(): File {
        //get the directory path and return the created file
        val storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile("image", "jpg", storageDir)

    }

    //get location with the maximum probability
    private fun getMax(array: FloatArray): Int {
        var index = 0
        var minimum = 0.0f
        for (i in 0..1000) {
            //if the value at the index is greater than minimum, store in index
            if (array[i] > minimum) {
                index = i
                minimum = array[i]
            }
        }
        return index
    }

    //save the state
    override fun onSaveInstanceState(outState: Bundle) {
        imageUri?.let { outState.putParcelable("imageUri", it) }
        label.let { outState.putString("label", it) }
        super.onSaveInstanceState(outState)
    }
}