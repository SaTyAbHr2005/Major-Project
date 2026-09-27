import React from 'react';
import { AppProvider, useApp } from './context/AppContext';
import { MLProvider } from './context/MLContext';

// Navigation Components
import { Sidebar } from './components/common/Sidebar';
import { TopNav } from './components/common/TopNav';
import { HospitalSidebar } from './components/common/HospitalSidebar';
import { HospitalTopNav } from './components/common/HospitalTopNav';

// Modals
import { HaltTrainingModal } from './components/modals/HaltTrainingModal';
import { QuarantineModal } from './components/modals/QuarantineModal';
import { ExportWeightsModal } from './components/modals/ExportWeightsModal';
import { SwitchRoleModal } from './components/modals/SwitchRoleModal';

// Auth & Homepage Views
import { HomeView } from './views/HomeView';
import { LoginView } from './views/LoginView';
import { RequestAccessView } from './views/RequestAccessView';
import { HospitalLoginView } from './views/HospitalLoginView';
import { HospitalRegisterView } from './views/HospitalRegisterView';

// Researcher / Admin Views
import { DashboardView } from './views/DashboardView';
import { HospitalsView } from './views/HospitalsView';
import { FederatedTrainingView } from './views/FederatedTrainingView';
import { ModelsView } from './views/ModelsView';
import { ExperimentsView } from './views/ExperimentsView';
import { MonitoringView } from './views/MonitoringView';
import { SecurityView } from './views/SecurityView';
import { AuditLogsView } from './views/AuditLogsView';
import { NotificationsView } from './views/NotificationsView';
import { MessagesView } from './views/MessagesView';
import { UserManagementView } from './views/UserManagementView';
import { SettingsView } from './views/SettingsView';
import { ProfileView } from './views/ProfileView';
import { StatesDemoView } from './views/StatesDemoView';
import { AdminApprovalView } from './views/AdminApprovalView';
import { AdminDiseaseModelsView } from './views/AdminDiseaseModelsView';

// Hospital Operator Views
import HospitalDashboardView from './views/HospitalDashboardView';
import HospitalDatasetView from './views/HospitalDatasetView';
import HospitalPreprocessingView from './views/HospitalPreprocessingView';
import HospitalStartTrainingView from './views/HospitalStartTrainingView';
import HospitalTrainingMonitorView from './views/HospitalTrainingMonitorView';
import HospitalTrainingResultView from './views/HospitalTrainingResultView';
import HospitalModelsView from './views/HospitalModelsView';
import HospitalInferenceView from './views/HospitalInferenceView';
import HospitalCommunicationView from './views/HospitalCommunicationView';
import HospitalSettingsView from './views/HospitalSettingsView';

const MainLayout = () => {
  const { activeScreen, isLoggedIn, userRole } = useApp();

  // 1. Unauthenticated state or explicitly requested Auth / Landing screens
  if (!isLoggedIn) {
    if (activeScreen === 'request-access') {
      return <RequestAccessView />;
    }
    if (activeScreen === 'hospital-register') {
      return <HospitalRegisterView />;
    }
    if (activeScreen === 'hospital-login') {
      return <HospitalLoginView />;
    }
    if (activeScreen === 'login') {
      return <LoginView />;
    }
    // Default entry point is HomeView
    return <HomeView />;
  }

  // If logged-in user explicitly visits home view
  if (activeScreen === 'home') {
    return <HomeView />;
  }

  // 2. Hospital Operator Layout & Views
  if (userRole === 'hospital_operator') {
    const renderHospitalScreen = () => {
      switch (activeScreen) {
        case 'dashboard':
          return <HospitalDashboardView />;
        case 'datasets':
          return <HospitalDatasetView />;
        case 'preprocessing':
          return <HospitalPreprocessingView />;
        case 'start_training':
          return <HospitalStartTrainingView />;
        case 'training_monitor':
          return <HospitalTrainingMonitorView />;
        case 'training_results':
          return <HospitalTrainingResultView />;
        case 'models':
          return <HospitalModelsView />;
        case 'inference':
          return <HospitalInferenceView />;
        case 'communication':
          return <HospitalCommunicationView />;
        case 'settings':
          return <HospitalSettingsView />;
        default:
          return <HospitalDashboardView />;
      }
    };

    return (
      <div style={{ display: 'flex', width: '100vw', height: '100vh', overflow: 'hidden' }}>
        <HospitalSidebar />
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
          <HospitalTopNav />
          <main style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
            {renderHospitalScreen()}
          </main>
        </div>
      </div>
    );
  }

  // 3. Researcher / Consortium Admin Layout & Views
  const renderResearcherScreen = () => {
    switch (activeScreen) {
      case 'dashboard':
        return <DashboardView />;
      case 'hospitals':
        return <HospitalsView />;
      case 'training':
        return <FederatedTrainingView />;
      case 'models':
        return <ModelsView />;
      case 'experiments':
        return <ExperimentsView />;
      case 'monitoring':
        return <MonitoringView />;
      case 'security':
        return <SecurityView />;
      case 'audit':
        return <AuditLogsView />;
      case 'notifications':
        return <NotificationsView />;
      case 'messages':
        return <MessagesView />;
      case 'users':
        return <UserManagementView />;
      case 'settings':
        return <SettingsView />;
      case 'profile':
        return <ProfileView />;
      case 'states':
        return <StatesDemoView />;
      case 'approvals':
        return <AdminApprovalView />;
      case 'disease-models':
        return <AdminDiseaseModelsView />;
      default:
        return <DashboardView />;
    }
  };

  return (
    <div style={{ display: 'flex', width: '100vw', height: '100vh', overflow: 'hidden' }}>
      <Sidebar />
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
        <TopNav />
        <main style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
          {renderResearcherScreen()}
        </main>
      </div>

      {/* Global Modals */}
      <HaltTrainingModal />
      <QuarantineModal />
      <ExportWeightsModal />
      <SwitchRoleModal />
    </div>
  );
};

export default function App() {
  return (
    <AppProvider>
      <MLProvider>
        <MainLayout />
      </MLProvider>
    </AppProvider>
  );
}
