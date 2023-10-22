"use client"
import { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { Holistic } from "@mediapipe/holistic";

export default function Home() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [sequence, setSequence] = useState([]);
  const [sentence, setSentence] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const threshold = 0.5;


  function drawResults(canvas, keypoints, sentence) {
    const ctx = canvas.getContext('2d');

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    keypoints.forEach(point => {
      ctx.beginPath();
      ctx.arc(point[0] * canvas.width, point[1] * canvas.height, 5, 0, 2 * Math.PI); 
      ctx.fillStyle = 'blue';
      ctx.fill();
    });

    ctx.fillStyle = '#F57710';
    ctx.fillRect(0, 0, canvas.width, 40);
    ctx.font = '24px Arial';
    ctx.fillStyle = 'white';
    ctx.fillText(sentence.join(' '), 10, 30);
  }
  
  useEffect(() => {
    async function loadModel() {
      const loadedModel = await tf.loadGraphModel('/graph_model/model.json');
      setModel(loadedModel);
    }

    loadModel();
  }, []);

  useEffect(() => {
    if (model === null) return;

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
          videoRef.current.srcObject = stream;
          videoRef.current.play();

          holistic.send({ image: videoRef.current });
        });
    }

    const holistic = new Holistic({locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1635989137/${file}`; 
    }});

    holistic.onResults(results => {
      console.log('RESULTS', results)
      const keypoints = extractKeyPointsFromResults(results);
      setSequence(prevSequence => {
        const newSequence = [...prevSequence, keypoints];
        return newSequence.slice(-30);
      });
      
      // Add prediction and visualization logic here
      if (sequence.length === 30) {
        const res = model.predict(tf.tensor(sequence).expandDims(0))[0];
        const actionIndex = res.argMax().dataSync()[0];
        console.log("RESPONSE", response)
        setPredictions(prevPredictions => [...prevPredictions, actionIndex]);

        if (predictions.slice(-10).every(val => val === actionIndex)) {
          if (res[actionIndex] > threshold) {
            setSentence(prevSentence => {
              if (prevSentence.length === 0 || actions[actionIndex] !== prevSentence[prevSentence.length - 1]) {
                return [...prevSentence, actions[actionIndex]].slice(-5);
              }
              return prevSentence;
            });
          }
        }

        drawResults(canvasRef.current, keypoints, sentence);
      }
    });

    
  }, [model]);

  function extractKeyPointsFromResults(results) {
    const pose = results.poseLandmarks ? results.poseLandmarks.map(res => [res.x, res.y, res.z, res.visibility]).flat() : new Array(33 * 4).fill(0);
    const face = results.faceLandmarks ? results.faceLandmarks.map(res => [res.x, res.y, res.z]).flat() : new Array(468 * 3).fill(0);
    const lh = results.leftHandLandmarks ? results.leftHandLandmarks.map(res => [res.x, res.y, res.z]).flat() : new Array(21 * 3).fill(0);
    const rh = results.rightHandLandmarks ? results.rightHandLandmarks.map(res => [res.x, res.y, res.z]).flat() : new Array(21 * 3).fill(0);

    return [...pose, ...face, ...lh, ...rh];
  }


  return (
    <div>
      <video ref={videoRef} width="640" height="480" />
      <canvas ref={canvasRef} width="640" height="480" />
    </div>
  );
}
