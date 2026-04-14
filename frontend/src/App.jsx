import React, { useState, useRef } from "react";
import { UploadCloud, Image as ImageIcon, X, Activity, Leaf, CheckCircle, AlertTriangle } from "lucide-react";

function App() {
    const [image, setImage] = useState(null);
    const [file, setFile] = useState(null);
    const [classification, setClassification] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [isDragging, setIsDragging] = useState(false);
    const fileInputRef = useRef(null);

    const handleFile = (selectedFile) => {
        if (selectedFile && (selectedFile.type === "image/jpeg" || selectedFile.type === "image/png")) {
            setFile(selectedFile);
            const imageURL = URL.createObjectURL(selectedFile);
            setImage(imageURL);
            setClassification(null);
        } else {
            alert("Please select only JPG or PNG files.");
        }
    };

    const onFileChange = (event) => {
        handleFile(event.target.files[0]);
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
            e.dataTransfer.clearData();
        }
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        if (!file) return;

        setIsLoading(true);
        setClassification(null);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch("http://localhost:8000/predict/", {
                method: 'POST',
                body: formData,
            });
            
            if (!response.ok) throw new Error("API Error");
            
            const data = await response.json();
            setClassification(data);
            setIsLoading(false);

        } catch (error) {
            console.log("Local server not found or error: ", error);
        }
    };

    const handleClearImage = () => {
        setImage(null);
        setFile(null);
        setClassification(null);
        setIsLoading(false);
        if (fileInputRef.current) fileInputRef.current.value = "";
    };

    const confidencePercentage = classification ? (classification.confidence * 100).toFixed(1) : 0;

    return (
        <div 
            className="min-h-screen bg-cover bg-center flex items-center justify-center p-4 font-sans"
            style={{ 
                backgroundImage: "url('https://images.unsplash.com/photo-1599940824399-b87987ceb72a?auto=format&fit=crop&q=80&w=2000')" 
            }}
        >
            <div className="absolute inset-0 bg-black/40 backdrop-blur-sm"></div>

            <div className="relative z-10 w-full max-w-lg bg-white/90 backdrop-blur-md rounded-3xl shadow-2xl border border-white/40 overflow-hidden flex flex-col">
                
                <div className="bg-gradient-to-r from-emerald-600 to-green-500 p-6 text-white text-center">
                    <div className="flex justify-center mb-2">
                        <Leaf className="w-10 h-10 text-emerald-100" />
                    </div>
                    <h1 className="text-2xl font-bold tracking-wide">Leaf Disease Prediction</h1>
                    <p className="text-emerald-100 text-sm mt-1">Corn (Zea mays)</p>
                </div>

                <form className="p-6 flex flex-col items-center w-full" onSubmit={handleSubmit}>
                    
                    <div className="w-full mb-6">
                        {image ? (
                            <div className="relative group w-full h-64 rounded-2xl overflow-hidden shadow-md">
                                <img src={image} alt="Preview" className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105" />
                                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
                                    <button 
                                        type="button" 
                                        onClick={handleClearImage}
                                        className="bg-red-500 text-white p-3 rounded-full hover:bg-red-600 transition-colors shadow-lg flex items-center gap-2 font-medium"
                                    >
                                        <X className="w-5 h-5" /> Remove
                                    </button>
                                </div>
                            </div>
                        ) : (
                            <div 
                                onClick={() => fileInputRef.current.click()}
                                onDragOver={handleDragOver}
                                onDragLeave={handleDragLeave}
                                onDrop={handleDrop}
                                className={`w-full h-64 border-3 border-dashed rounded-2xl flex flex-col items-center justify-center cursor-pointer transition-all duration-300 ${
                                    isDragging 
                                    ? "border-emerald-500 bg-emerald-50 scale-[1.02]" 
                                    : "border-gray-300 bg-gray-50/50 hover:border-emerald-400 hover:bg-emerald-50/30"
                                }`}
                            >
                                <input 
                                    type="file" 
                                    ref={fileInputRef} 
                                    onChange={onFileChange} 
                                    accept=".jpg,.jpeg,.png" 
                                    className="hidden" 
                                />
                                <div className={`p-4 rounded-full mb-3 ${isDragging ? 'bg-emerald-100 text-emerald-600' : 'bg-gray-100 text-gray-400'}`}>
                                    <UploadCloud className="w-10 h-10" />
                                </div>
                                <p className="text-gray-700 font-semibold text-lg text-center">
                                    Drag your image or <span className="text-emerald-600">click here</span>
                                </p>
                                <p className="text-gray-400 text-sm mt-2">Supports JPG or PNG</p>
                            </div>
                        )}
                    </div>

                    <div className="flex gap-3 w-full">
                        <button 
                            type="button" 
                            onClick={handleClearImage} 
                            disabled={!image || isLoading}
                            className="flex-1 py-3 px-4 rounded-xl font-semibold text-gray-600 bg-gray-100 hover:bg-gray-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                        >
                            Clear
                        </button>
                        <button 
                            type="submit" 
                            disabled={!image || isLoading}
                            className="flex-[2] py-3 px-4 rounded-xl font-bold text-white bg-gradient-to-r from-emerald-500 to-green-600 hover:from-emerald-600 hover:to-green-700 shadow-md hover:shadow-lg transform active:scale-95 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                        >
                            {isLoading ? (
                                <>
                                    <Activity className="w-5 h-5 animate-pulse" /> Analyzing...
                                </>
                            ) : (
                                <>
                                    <ImageIcon className="w-5 h-5" /> Analyze Image
                                </>
                            )}
                        </button>
                    </div>

                    {isLoading && (
                        <div className="mt-6 flex flex-col items-center justify-center animate-in fade-in zoom-in duration-300">
                            <div className="w-10 h-10 border-4 border-emerald-200 border-t-emerald-600 rounded-full animate-spin"></div>
                            <p className="text-emerald-700 font-medium mt-3 animate-pulse">Processing with AI...</p>
                        </div>
                    )}

                    {classification && !isLoading && (
                        <div className="w-full mt-6 p-5 bg-white border border-emerald-100 rounded-2xl shadow-sm animate-in slide-in-from-bottom-4 duration-500">
                            <h3 className="text-gray-500 text-xs font-bold uppercase tracking-wider mb-4 border-b pb-2 flex items-center gap-2">
                                <CheckCircle className="w-4 h-4 text-emerald-500" />
                                Analysis Result
                            </h3>
                            
                            <div className="mb-4">
                                <p className="text-sm text-gray-500 mb-1">Detected Classification:</p>
                                <p className="text-xl font-bold text-gray-800 flex items-center gap-2">
                                    {classification.class === "Healthy" ? (
                                        <Leaf className="w-5 h-5 text-green-500" />
                                    ) : (
                                        <AlertTriangle className="w-5 h-5 text-amber-500" />
                                    )}
                                    {classification.class}
                                </p>
                            </div>

                            <div>
                                <div className="flex justify-between items-end mb-1">
                                    <p className="text-sm text-gray-500">Confidence Level:</p>
                                    <p className="text-lg font-bold text-emerald-600">{confidencePercentage}%</p>
                                </div>
                                <div className="w-full bg-gray-100 rounded-full h-3 overflow-hidden">
                                    <div 
                                        className="bg-gradient-to-r from-emerald-400 to-green-600 h-3 rounded-full transition-all duration-1000 ease-out" 
                                        style={{ width: `${confidencePercentage}%` }}
                                    ></div>
                                </div>
                            </div>
                        </div>
                    )}
                </form>
            </div>
        </div>
    );
}

export default App;