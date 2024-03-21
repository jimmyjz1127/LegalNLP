import React from 'react';
import Plot from 'react-plotly.js';
import { useState } from "react";
import { useEffect } from "react";

const BarChart = (props) => {

	const {data} = props;


	// Assuming `data` is your object where keys are x values and values are the bar heights
	const keys = Object.keys(data);
	const values = Object.values(data);

	return (
		<div>
			<Plot
				data={[
					{
						type: 'bar',
						x: values.map(([key,val]) => key),
						y: values.map(([key,val]) => val),
					},
				]}
				layout={{
					width: 620,
					height: 400,
					title: 'Top 10 Tokens by Attention',
					xaxis: { 
						title: 'Token',
						tickmode : 'array',
						ticktext : keys,

					},
					yaxis: { 
						title: 'Attention Score',
						tickmode : 'array'
					},
				}}
			/>
		</div>
	);
};

export default BarChart;
