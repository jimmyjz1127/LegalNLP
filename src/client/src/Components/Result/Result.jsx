import React, { useLayoutEffect } from "react";
import { useState } from "react";
import { useEffect } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Axios from 'axios';

import "./Result.css"

import EyeIcon from './../../Assets/eye.png'

function Result(props){
    const {data, index} = props;

    const [showDoc, setShowDoc] = useState(false);

    return (
        <div className="result-wrapper flex col align-center">
            <div className="result">
                <h3>{index + 1}</h3>
                <div className="result-info flex col align-start justify-center">
                    <text className="result-name">{data.name}</text>
                    <text>Court : {data.court}</text>
                    <text>Similarity Score : <text className="highlight">{data.similarity}</text></text>
                </div>
                <button className="view-btn" onClick={(e) => setShowDoc(!showDoc)}>
                    View
                </button>
            </div>
            {showDoc &&
                <div className="doc-viewer">
                    {data.main}
                </div>
            }
            
        </div>
    )
}

export default Result;