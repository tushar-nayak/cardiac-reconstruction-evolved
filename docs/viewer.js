async function loadViewer() {
  const target = document.getElementById("viewer");
  const response = await fetch("assets/reconstruction_mesh.json");
  const mesh = await response.json();

  const trace = {
    type: "mesh3d",
    x: mesh.x,
    y: mesh.y,
    z: mesh.z,
    i: mesh.i,
    j: mesh.j,
    k: mesh.k,
    color: "#0057b8",
    opacity: 0.72,
    flatshading: false,
    lighting: {
      ambient: 0.46,
      diffuse: 0.78,
      roughness: 0.72,
      specular: 0.18
    },
    lightposition: { x: 120, y: 80, z: 180 },
    hovertemplate: "x=%{x:.1f}<br>y=%{y:.1f}<br>z=%{z:.1f}<extra></extra>"
  };

  const layout = {
    margin: { l: 0, r: 0, t: 0, b: 0 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    scene: {
      aspectmode: "data",
      xaxis: { title: "x", showbackground: false, gridcolor: "#d8d1c3" },
      yaxis: { title: "y", showbackground: false, gridcolor: "#d8d1c3" },
      zaxis: { title: "z", showbackground: false, gridcolor: "#d8d1c3" },
      camera: { eye: { x: 1.45, y: 1.25, z: 0.9 } }
    }
  };

  const config = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"]
  };

  Plotly.newPlot(target, [trace], layout, config);
}

loadViewer().catch((error) => {
  const target = document.getElementById("viewer");
  target.textContent = `Viewer failed to load: ${error.message}`;
});
