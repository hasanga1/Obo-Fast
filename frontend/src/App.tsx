import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import ManageFiles from './pages/ManageFiles';
import UploadFiles from './pages/UploadFiles';
import SideBar from './components/SideBar';
import { Box } from '@mui/material';  // Import Box for layout styling

function App() {
  return (
    <Router>
      <Box sx={{ display: 'flex' }}>  {/* Flex container for sidebar and content */}
        <SideBar />
        <Box sx={{ flexGrow: 1, padding: 3 }}>  {/* Content area with padding */}
          <Routes>
            <Route path="/" element={<UploadFiles />} /> {/* Default Route */}
            <Route path="/upload" element={<UploadFiles />} /> {/* Route for UploadFiles */}
            <Route path="/manage" element={<ManageFiles />} /> {/* Route for ManageFiles */}
          </Routes>
        </Box>
      </Box>
    </Router>
  );
}

export default App;
