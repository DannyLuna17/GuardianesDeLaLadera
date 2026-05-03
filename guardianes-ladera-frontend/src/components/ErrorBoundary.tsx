import { Component, type ReactNode } from 'react'

interface ErrorBoundaryProps {
  children: ReactNode
}

interface ErrorBoundaryState {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error }
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null })
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="page-state">
          <div className="page-state-card error">
            <div className="page-state-title">Ocurrio un error inesperado</div>
            <div className="page-state-text">
              {this.state.error?.message ?? 'Error desconocido'}
            </div>
            <button className="primary" onClick={this.handleRetry}>
              Reintentar
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
