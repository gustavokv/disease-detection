import { FileUploader } from "react-drag-drop-files";
import { TailSpin } from "react-loader-spinner";
import { useState } from "react";
import axios from "axios";

const fileTypes = ["JPG", "PNG"];

function App() {
    const [image, setImage] = useState(null);
    const [file, setFile] = useState(null);
    const [classification, setClassification] = useState(null);
    const [isLoading, setIsLoading] = useState(false);

    const handleFileChange = (file) => {
        setFile(file);
        const imageURL = URL.createObjectURL(file); 
        setImage(imageURL);
    }

    const handleSubmit = async (event) => {
        event.preventDefault();
        let response = null;
        const formData = new FormData();
        formData.append('file', file); // 'file' deve ser o nome esperado no backend

        setIsLoading(true);
        
        try{
            response = await axios.post("http://localhost:8000/predict/", formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });
        } catch(error) {
            console.log(error);
        }

        if(response) setClassification(response.data);

        setIsLoading(false);
    }

    const handleClearImage = () => {
        setImage(null);
        setClassification(null);
        setIsLoading(false);
    }

    return (
        <div className="h-screen bg-[url('./assets/background.jpg')]">
            <div className="h-screen flex items-center justify-center backdrop-blur-xs">
                <form name="file" className="pb-5 pt-5 font-mono flex flex-col items-center justify-center isolate aspect-video w-96 rounded-4xl bg-white/20 shadow-lg ring-1 ring-black/5" onSubmit={(e) => handleSubmit(e)}>
                    {image ? 
                        <img src={image} className="h-80 w-80 rounded-2xl shadow-xl/30 mb-5"/>
                        :
                        <FileUploader 
                            handleChange={handleFileChange} 
                            name="file" 
                            types={fileTypes} 
                            required
                        >
                            <div className="border-3 border-dashed rounded-3xl flex flex-col items-center text-2xl py-5 px-3 my-6 cursor-pointer opacity-50 text-white hover:scale-105 transform transition duration-0.3">
                                <p className="text-6xl mb-1">📁</p>
                                <p><b>Arraste ou clique aqui</b></p>
                                <p><b>para selecionar a imagem</b></p>
                            </div>
                        </FileUploader>
                    }

                    <div>
                        <button type="submit" className="text-white bg-gradient-to-r from-green-400 via-green-500 to-green-600 hover:bg-gradient-to-br text-lg rounded-3xl px-5 py-2.5 text-center me-2 mb-2 cursor-pointer hover:scale-108 transform transition duration-0.2"><b>ENVIAR</b></button>
                        <button onClick={handleClearImage} className="text-white bg-gradient-to-r from-green-400 via-green-500 to-green-600 hover:bg-gradient-to-br text-lg rounded-3xl px-5 py-2.5 text-center me-2 mb-2 cursor-pointer hover:scale-108 transform transition duration-0.2"><b>LIMPAR</b></button>
                    </div>
                    
                    <div className="mt-5">
                        {isLoading ? 
                            <TailSpin
                                color="#0fd850"
                                ariaLabel="tail-spin-loading"
                                radius="1"
                                wrapperStyle={{}}
                                wrapperClass=""
                                width="50"
                            />
                            :
                            null
                        }
                    </div>

                    {classification ? 
                        <div className="flex text-gray-200 mt-5">
                            <div className="mr-10"> 
                                <p className="text-md">Classificação:</p>
                                <p className="text-xl"><b>{classification.class}</b></p>
                            </div>

                            <div>
                                <p className="text-md">Confiança:</p>
                                <p className="text-xl"><b>{classification.confidence.toFixed(2) * 100}%</b></p>
                            </div>
                        </div>
                        :
                        null
                    }
                </form>
            </div>
        </div>
    )
}

export default App