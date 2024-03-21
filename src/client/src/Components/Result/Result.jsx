import React, { useLayoutEffect } from "react";
import { useState } from "react";
import { useEffect } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Axios from 'axios';

import AttentionVisual from "./AttentionVisual";

import "./Result.css"

import EyeIcon from './../../Assets/eye.png'
import InsightIcon from './../../Assets/insight.png'

function Result(props){
    const {data, index} = props;

    const [showDoc, setShowDoc] = useState(false);
    const [showInsights, setShowInsights] = useState(false);

    return (
        <div className="result-wrapper flex col align-center">
            <div className="result">
                <h3>{index + 1}</h3>
                <div className="result-info flex col align-start justify-center">
                    <text className="result-name">{data.name}</text>
                    <text>Court : {data.court}</text>
                    <text>Similarity Score : <text className="highlight">{data.similarity}</text></text>
                </div>
                <div className="flex col align-end">
                    <button className="view-btn" onClick={(e) => setShowDoc(!showDoc)}>
                        Document <img src={EyeIcon}/>
                    </button>
                    <button className="view-btn" onClick={(e) => setShowInsights(!showInsights)}>
                        Insights <img src={InsightIcon}/>
                    </button>
                </div>
            </div>
            {showInsights &&
                <AttentionVisual attentionData={JSON.parse(data.attention)} query={JSON.parse(data.query_tokens)}/>
            }
            {showDoc &&
                <div className="doc-viewer">
                    {data.main}
                </div>
            }
            
        </div>
    )
}

export default Result;