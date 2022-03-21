package com.example.imagerecognition

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import com.example.imagerecognition.ml.MobilenetV110224Quant
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {

    //declare the variables
    lateinit var image: ImageView
    lateinit var select: Button
    lateinit var predict: Button
    lateinit var text: TextView
    lateinit var bitmap: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        //initialise the view variables
        image = findViewById(R.id.image)
        select = findViewById(R.id.selectButton)
        predict = findViewById(R.id.predictButton)
        text = findViewById(R.id.text)

        //get the labels from the textfile
        val fileName = "labels.txt"
        val input = application.assets.open(fileName).bufferedReader().use {it.readText()}
        var labelList = input.split("\n")


        //set onclicklistener to select an image when select button is pressed
        select.setOnClickListener(View.OnClickListener {

            //clear the textview
            text.setText(" ")

            //create the intent
            var intent: Intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "image/*"

            //start the activity for the intent
            startActivityForResult(intent, 100)
        })

        predict.setOnClickListener(View.OnClickListener {
            //resize the image to be fed into the model
            var resizedBitmap: Bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

            val model = MobilenetV110224Quant.newInstance(this)

            // Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)

            //create bytebuffer from the resized bitmap
            var tensorBuffer = TensorImage.fromBitmap(resizedBitmap)
            var byteBuffer = tensorBuffer.buffer

            inputFeature0.loadBuffer(byteBuffer)

            // Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer

            //get correct label for the prediction
            var max = getMax(outputFeature0.floatArray)

            //show the prediction in the textview
            text.setText(labelList[max])

            // Releases model resources if no longer used.
            model.close()

        })

    }
    //once the user selects the image, set the image in imageview and store to bitmap
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        //set image in imageview
        image.setImageURI(data?.data)

        //convert to bitmap
        var uri: Uri? = data?.data
        bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)

    }

    //get location with the maximum probability
    fun getMax(array:FloatArray): Int {
       var index = 0
       var minimum = 0.0f

       for(i in 0..1000) {
           //if the value at the index is greater than minimum, store in index
           if (array[i]>minimum) {
               index = i
               minimum = array[i]
           }
       }
        return index
    }

}