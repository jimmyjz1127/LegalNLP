import React, { useLayoutEffect } from "react";
import { useState } from "react";
import { useEffect } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Axios from 'axios';

import "./Result.css"

function Result(props){
    const {data, index} = props;

    return (
        <div className="result-wrapper flex row align-center">
            <div className="result flex row align-center">
                <h2>{index + 1}</h2>
                <div className="result-name">{data.name}</div>
            </div>
        </div>
    )
}

export default Result;