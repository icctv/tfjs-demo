import React from 'react'
import {
  Text,
  View,
  ActivityIndicator,
  StatusBar,
  Image,
  TouchableOpacity
} from 'react-native'
import { Camera } from 'expo-camera'
import { styles } from './styles'
import * as ImageManipulator from 'expo-image-manipulator'
import * as tf from '@tensorflow/tfjs'
import { fetch } from '@tensorflow/tfjs-react-native'
import * as mobilenet from '@tensorflow-models/mobilenet'
import * as jpeg from 'jpeg-js'
import * as ImagePicker from 'expo-image-picker'
import Constants from 'expo-constants'
import * as Permissions from 'expo-permissions'

export async function base64ImageToTensor(base64) {
  const rawImageData = tf.util.encodeString(base64, 'base64')
  const TO_UINT8ARRAY = true
  const {width, height, data} = jpeg.decode(rawImageData, TO_UINT8ARRAY)
  // Drop the alpha channel info
  const buffer = new Uint8Array(width * height * 3)
  let offset = 0  // offset into original data
  for (let i = 0; i < buffer.length; i += 3) {
    buffer[i] = data[offset]
    buffer[i + 1] = data[offset + 1]
    buffer[i + 2] = data[offset + 2]

    offset += 4
  }
  return tf.tensor3d(buffer, [height, width, 3])
}

class App extends React.Component {
  state = {
    isTfReady: false,
    isModelReady: false,
    isCameraActive: false,
    predictions: null,
    image: null
  }

  constructor (props) {
    super(props)
    this.toggleCamera = this.toggleCamera.bind(this)
    this.classify = this.classify.bind(this)
  }

  async toggleCamera() {
    this.setState({ isCameraActive: !this.state.isCameraActive })
  }


  async componentDidMount() {
    await tf.ready()
    this.setState({
      isTfReady: true
    })
    console.log('Loading mobilenet')
    await this.getPermissionAsync()
    setTimeout(this.toggleCamera, 100)

    this.model = await mobilenet.load({
      version: 2,
      alpha: 0.75 // 0.25 / 0.50 / 0.75 / 1
    })
    this.setState({ isModelReady: true })
    console.log('Loaded mobilenet')

    setTimeout(this.classify, 100)
  }

  getPermissionAsync = async () => {
    if (Constants.platform.ios) {
      const { status } = await Permissions.askAsync(Permissions.CAMERA)
      if (status !== 'granted') {
        alert('Sorry, we need camera permissions to make this work!')
      }
    }
  }

  async classify () {
    if (this.camera && this.state.isCameraActive && this.state.isModelReady) {
      await this.classifyImage()
    }

    setTimeout(this.classify, 10)
  }

  imageToTensor(rawImageData) {
    const TO_UINT8ARRAY = true
    const { width, height, data } = jpeg.decode(rawImageData, TO_UINT8ARRAY)
    // Drop the alpha channel info for mobilenet
    const buffer = new Uint8Array(width * height * 3)
    let offset = 0 // offset into original data
    for (let i = 0; i < buffer.length; i += 3) {
      buffer[i] = data[offset]
      buffer[i + 1] = data[offset + 1]
      buffer[i + 2] = data[offset + 2]

      offset += 4
    }

    return tf.tensor3d(buffer, [height, width, 3])
  }

  classifyImage = async () => {
    try {
      const maxWidth = 100

      const totalStart = new Date()

      // Take
      let start = new Date()
      const picture = await this.camera.takePictureAsync({
        quality: 0.3,
        skipProcessing: true,
        fixOrientation: false,
        exif: false
      })
      let end = new Date()
      const msPicture = end - start

      // Scale
      start = new Date()
      const scaled = await ImageManipulator.manipulateAsync(
        picture.uri,
        [{ resize: { width: maxWidth } }],
        { compress: 0.3, format: ImageManipulator.SaveFormat.JPEG })
      end = new Date()
      const msScale = end - start

      // Load
      start = new Date()
      const response = await fetch(scaled.uri, {}, { isBinary: true })
      const rawImageData = await response.arrayBuffer()
      const imageTensor = this.imageToTensor(rawImageData)
      end = new Date()
      const msLoad = end - start

      // Classify
      start = new Date()
      const predictions = await this.model.classify(imageTensor)
      end = new Date()
      const msClassify = end - start

      const result = {
        predictions,
        timings: {
          total: msTotal,
          picture: msPicture,
          scale: msScale,
          load: msLoad,
          classify: msClassify,
        }
      }

      const totalEnd = new Date()
      const msTotal = totalEnd - totalStart

      this.setState(result)

      // Clear console

      console.log(`
\x1Bc'
${msTotal} ms total
~${(1000 / msTotal).toFixed(2)} fps
---
${msPicture} ms to take picture
${msScale} ms to scale to ${maxWidth}px
${msLoad} ms to load
${msClassify} ms to classify
---
${predictions.map(p => `${(p.probability * 100).toFixed(2)}% - ${p.className}`).join('\n')}
      `)

      return result
    } catch (error) {
      console.log(error)
    }
  }

  selectImage = async () => {
    try {
      let response = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.All,
        allowsEditing: true,
        aspect: [4, 3]
      })

      if (!response.cancelled) {
        const source = { uri: response.uri }
        this.setState({ image: source })
        this.classifyImage()
      }
    } catch (error) {
      console.log(error)
    }
  }

  render() {
    const { isModelReady, isCameraActive, predictions, image, timings } = this.state

    return (
      <TouchableOpacity
        style={styles.container}
        onPress={this.toggleCamera}>
        <StatusBar barStyle='light-content' />
        <View style={styles.loadingContainer}>
          <View style={styles.loadingModelContainer}>
            <Text style={styles.text}>Model ready? </Text>
            {isModelReady ? (
              <Text style={styles.text}>✅</Text>
            ) : (
              <ActivityIndicator size='small' />
            )}
          </View>
        </View>

        <View style={styles.loadingModelContainer}>
          <Text style={styles.text}>
            Camera active? {isCameraActive ? <Text>✅</Text> : ''}
          </Text>
        </View>

        <Camera
          ref={ref => {
            this.camera = ref
          }}
          style={{ width: 300, height: 300 }}
          pictureSize="640x480"
        >
          <View
            style={{
              flex: 1,
              backgroundColor: 'transparent',
              flexDirection: 'row',
            }}>
              <Text style={styles.text}>Yolo</Text>
          </View>
        </Camera>

        {/* <TouchableOpacity
          style={styles.imageWrapper}
          onPress={isModelReady ? this.selectImage : undefined}>
          {image && <Image source={image} style={styles.imageContainer} />}

          {isModelReady && !image && (
            <Text style={styles.transparentText}>Tap to choose image</Text>
          )}
        </TouchableOpacity> */}
        <View style={styles.predictionWrapper}>
          {isModelReady && image && (
            <Text style={styles.text}>
              Predictions: {predictions ? '' : 'Predicting...'}
            </Text>
          )}
          {isModelReady &&
            predictions &&
            predictions.map(p =>
              <Text key={p.className} style={styles.text}>
                {(p.probability * 100).toFixed(2)}% - {p.className}
              </Text>
            )}
          {isModelReady && predictions && timings && Object.keys(timings).map(k =>
            <Text key={k} style={styles.text}>{timings[k]} ms to {k}</Text>
          )}
        </View>
      </TouchableOpacity>
    )
  }
}


export default App
