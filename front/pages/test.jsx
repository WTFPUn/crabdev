import React, { useEffect, useState } from 'react';

const TestPage = () => {
  // strcture of image is {image: string, format: string, molt: boolean}
  const [images, setImages] = useState([]);

  useEffect(() => {
    // Connect to the WebSocket (localhost:8000)
    const socket = new WebSocket('ws://localhost:8000/ws');

    socket.onopen = () => {
      console.log('WebSocket connection established');
    };

    // Handle incoming messages
    socket.onmessage = async (event) => {
      const data = JSON.parse(JSON.parse(event.data));
      // convert string to json
      
      // get data keyname
      // console.log(data);

      if (data === null) return;

      if (data.type === "image_collection") {
        setImages(data.data);
        console.log(Object.keys(data.data[0]))
      }

    };

    // Clean up the WebSocket connection
    return () => {
      socket.close();
    };
  }, []);

  return (
    <div>
      {/* <h1>Test Page</h1> */}
      {/* <p>Received message: {message}</p> */}
      <ImageCollection images={images} />
    </div>
  );
};

export default TestPage;

const ImageCollection = ({ images }) => {
  // console.log(images[0].molt);
  return (
    <div className='w-full flex flex-wrap gap-4'>
      {images.map((image, index) => (
        <div className='w-[25rem]  flex flex-col' key={index}>
        <img src={`data:image/${image.format};base64,${image.image}`} alt={`image-${index}`}  className='w-full h-1/2' />
        <div className='text-white'>{image.molt ? 'Molt' : 'Not Molt'}</div>
       </div> 
      ))}
    </div>
  );
}
