// Variáveis globais
let currentResults = null;
let rewardChart = null;
let regretChart = null;
let rewardDistributionChart = null;
let rewardVsRegretChart = null;

// Inicialização
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    loadSampleData();
});

// Event Listeners
function initializeEventListeners() {
    // Data source radio buttons
    document.querySelectorAll('input[name="data_source"]').forEach(radio => {
        radio.addEventListener('change', function() {
            const uploadSection = document.getElementById('upload-section');
            if (this.value === 'upload') {
                uploadSection.style.display = 'block';
            } else {
                uploadSection.style.display = 'none';
                loadSampleData();
            }
        });
    });

    // Range sliders
    document.getElementById('generations').addEventListener('input', function() {
        document.getElementById('generations-value').textContent = this.value;
    });

    document.getElementById('iterations').addEventListener('input', function() {
        document.getElementById('iterations-value').textContent = this.value;
    });
}

// Carregar dados de exemplo
function loadSampleData() {
    // Simular carregamento de dados de exemplo
    const controlPreview = document.getElementById('control-preview');
    const testPreview = document.getElementById('test-preview');
    
    controlPreview.innerHTML = `
        <table>
            <tr><th>Campaign Name</th><th>Date</th><th>Spend [USD]</th><th># of Purchase</th></tr>
            <tr><td>Control Campaign</td><td>1.08.2019</td><td>2280</td><td>618</td></tr>
            <tr><td>Control Campaign</td><td>2.08.2019</td><td>1757</td><td>511</td></tr>
            <tr><td>Control Campaign</td><td>3.08.2019</td><td>2343</td><td>372</td></tr>
            <tr><td>...</td><td>...</td><td>...</td><td>...</td></tr>
        </table>
    `;
    
    testPreview.innerHTML = `
        <table>
            <tr><th>Campaign Name</th><th>Date</th><th>Spend [USD]</th><th># of Purchase</th></tr>
            <tr><td>Test Campaign</td><td>1.08.2019</td><td>2450</td><td>720</td></tr>
            <tr><td>Test Campaign</td><td>2.08.2019</td><td>1890</td><td>580</td></tr>
            <tr><td>Test Campaign</td><td>3.08.2019</td><td>2560</td><td>420</td></tr>
            <tr><td>...</td><td>...</td><td>...</td><td>...</td></tr>
        </table>
    `;
}

// Upload de arquivos
async function uploadFiles() {
    const controlFile = document.getElementById('control-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    
    if (!controlFile || !testFile) {
        showMessage('Por favor, selecione ambos os arquivos CSV.', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('control_file', controlFile);
    formData.append('test_file', testFile);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            showMessage('Arquivos carregados com sucesso!', 'success');
            // Aqui você pode adicionar preview dos dados carregados
        } else {
            showMessage(result.error, 'error');
        }
    } catch (error) {
        showMessage('Erro ao fazer upload dos arquivos.', 'error');
    }
}

// Executar experimento
async function runExperiment() {
    const algorithm = document.getElementById('algorithm').value;
    const optimizer = document.getElementById('optimizer').value;
    const generations = document.getElementById('generations').value;
    const iterations = document.getElementById('iterations').value;
    const useSampleData = document.querySelector('input[name="data_source"]:checked').value === 'sample';
    
    // Mostrar progresso
    const progressSection = document.getElementById('progress-section');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    
    progressSection.style.display = 'block';
    progressFill.style.width = '0%';
    progressText.textContent = '🔄 Preparando experimento...';
    
    // Simular progresso
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;
        progressFill.style.width = progress + '%';
        progressText.textContent = `🔄 Executando... ${Math.round(progress)}%`;
    }, 500);
    
    try {
        const response = await fetch('/run_experiment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                algorithm: algorithm,
                optimizer: optimizer,
                n_generations: parseInt(generations),
                n_iterations: parseInt(iterations),
                use_sample_data: useSampleData
            })
        });
        
        const result = await response.json();
        
        clearInterval(progressInterval);
        progressFill.style.width = '100%';
        progressText.textContent = '✅ Concluído!';
        
        if (result.success) {
            currentResults = result.results;
            displayResults(result);
            showMessage('🎉 Experimento concluído com sucesso!', 'success');
        } else {
            showMessage(result.error, 'error');
        }
        
        setTimeout(() => {
            progressSection.style.display = 'none';
        }, 2000);
        
    } catch (error) {
        clearInterval(progressInterval);
        progressSection.style.display = 'none';
        showMessage('❌ Erro ao executar experimento.', 'error');
    }
}

// Exibir resultados
function displayResults(result) {
    const resultsSection = document.getElementById('results-section');
    const noResults = document.getElementById('no-results');
    
    resultsSection.style.display = 'block';
    noResults.style.display = 'none';
    
    // Exibir configuração do experimento
    displayExperimentConfig(result.experiment_config);
    
    // Exibir estatísticas dos dados
    displayDataStatistics(result.data_statistics);
    
    // Exibir métricas de treinamento
    displayTrainingMetrics(result.training_metrics);
    
    // Exibir hiperparâmetros otimizados
    displayOptimizedParameters(result.optimized_parameters, result.detailed_results);
    
    // Atualizar resumo
    document.getElementById('total-runs').textContent = result.summary.total_runs;
    document.getElementById('avg-reward').textContent = result.summary.avg_reward.toFixed(2);
    document.getElementById('best-reward').textContent = result.summary.best_reward.toFixed(2);
    document.getElementById('worst-reward').textContent = result.summary.worst_reward.toFixed(2);
    document.getElementById('avg-regret').textContent = result.summary.avg_regret.toFixed(2);
    document.getElementById('best-regret').textContent = result.summary.best_regret.toFixed(2);
    
    // Criar tabela de resultados
    createResultsTable(result.results);
    
    // Criar gráficos
    createCharts(result.results);
}

// Exibir configuração do experimento
function displayExperimentConfig(config) {
    const section = document.getElementById('experiment-config-section');
    section.style.display = 'block';
    
    document.getElementById('exp-algorithm').textContent = config.algorithm;
    document.getElementById('exp-optimizer').textContent = config.optimizer;
    document.getElementById('exp-generations').textContent = config.n_generations;
    document.getElementById('exp-iterations').textContent = config.n_iterations;
    document.getElementById('exp-data-source').textContent = config.data_source;
    document.getElementById('exp-reward-metric').textContent = config.reward_metric;
    document.getElementById('exp-optimization-goal').textContent = config.optimization_goal;
}

// Exibir estatísticas dos dados
function displayDataStatistics(stats) {
    const section = document.getElementById('data-stats-section');
    section.style.display = 'block';
    
    document.getElementById('total-campaigns').textContent = stats.total_campaigns;
    document.getElementById('total-days').textContent = stats.total_days;
    document.getElementById('total-spend').textContent = `$${stats.total_spend.toLocaleString()}`;
    document.getElementById('avg-purchases').textContent = stats.avg_purchases.toFixed(2);
    document.getElementById('avg-roi').textContent = `${(stats.avg_roi * 100).toFixed(2)}%`;
    document.getElementById('conversion-rate').textContent = `${(stats.conversion_rate * 100).toFixed(3)}%`;
}

// Exibir métricas de treinamento
function displayTrainingMetrics(metrics) {
    const section = document.getElementById('training-metrics-section');
    section.style.display = 'block';
    
    document.getElementById('total-iterations').textContent = metrics.total_iterations;
    document.getElementById('successful-runs').textContent = metrics.successful_runs;
    document.getElementById('failed-runs').textContent = metrics.failed_runs;
    document.getElementById('convergence-rate').textContent = `${(metrics.convergence_rate * 100).toFixed(1)}%`;
    document.getElementById('std-reward').textContent = metrics.std_reward.toFixed(2);
    document.getElementById('std-regret').textContent = metrics.std_regret.toFixed(2);
    document.getElementById('reward-improvement').textContent = metrics.reward_improvement.toFixed(2);
    document.getElementById('regret-reduction').textContent = metrics.regret_reduction.toFixed(2);
}

// Exibir hiperparâmetros otimizados
function displayOptimizedParameters(params, detailedResults) {
    const section = document.getElementById('optimized-params-section');
    section.style.display = 'block';
    
    // Melhores parâmetros
    const optimizedParamsDiv = document.getElementById('optimized-parameters');
    if (Object.keys(params).length > 0) {
        optimizedParamsDiv.innerHTML = Object.entries(params)
            .map(([key, value]) => `<div><strong>${key}:</strong> ${typeof value === 'number' ? value.toFixed(4) : value}</div>`)
            .join('');
    } else {
        optimizedParamsDiv.innerHTML = '<div>Nenhum parâmetro otimizado disponível</div>';
    }
    
    // Melhor execução
    const bestRunDiv = document.getElementById('best-run-details');
    const bestRun = detailedResults.best_run;
    bestRunDiv.innerHTML = `
        <div><strong>Iteração:</strong> ${bestRun.iteration}</div>
        <div><strong>Recompensa:</strong> ${bestRun.reward.toFixed(2)}</div>
        <div><strong>Arrependimento:</strong> ${bestRun.regret.toFixed(2)}</div>
        <div><strong>Fitness:</strong> ${bestRun.fitness.toFixed(2)}</div>
    `;
    
    // Pior execução
    const worstRunDiv = document.getElementById('worst-run-details');
    const worstRun = detailedResults.worst_run;
    worstRunDiv.innerHTML = `
        <div><strong>Iteração:</strong> ${worstRun.iteration}</div>
        <div><strong>Recompensa:</strong> ${worstRun.reward.toFixed(2)}</div>
        <div><strong>Arrependimento:</strong> ${worstRun.regret.toFixed(2)}</div>
        <div><strong>Fitness:</strong> ${worstRun.fitness.toFixed(2)}</div>
    `;
}

// Criar tabela de resultados
function createResultsTable(results) {
    const tableContainer = document.getElementById('results-table');
    
    if (results.length === 0) {
        tableContainer.innerHTML = '<p>Nenhum resultado disponível.</p>';
        return;
    }
    
    let tableHTML = '<table><thead><tr>';
    
    // Cabeçalhos
    const headers = Object.keys(results[0]);
    headers.forEach(header => {
        const displayName = header.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        tableHTML += `<th>${displayName}</th>`;
    });
    tableHTML += '</tr></thead><tbody>';
    
    // Dados
    results.forEach((row, index) => {
        tableHTML += '<tr>';
        headers.forEach(header => {
            let value = row[header];
            if (typeof value === 'number') {
                value = value.toFixed(2);
            } else if (typeof value === 'object' && value !== null) {
                value = JSON.stringify(value);
            }
            tableHTML += `<td>${value}</td>`;
        });
        tableHTML += '</tr>';
    });
    
    tableHTML += '</tbody></table>';
    tableContainer.innerHTML = tableHTML;
}

// Criar gráficos
function createCharts(results) {
    if (results.length === 0) return;
    
    // Destruir gráficos existentes
    if (rewardChart) rewardChart.destroy();
    if (regretChart) regretChart.destroy();
    if (rewardDistributionChart) rewardDistributionChart.destroy();
    if (rewardVsRegretChart) rewardVsRegretChart.destroy();
    
    const labels = results.map((_, index) => `Iteração ${index + 1}`);
    const rewards = results.map(r => r.total_reward);
    const regrets = results.map(r => r.total_regret);
    
    // Gráfico de recompensas
    const rewardCtx = document.getElementById('reward-chart').getContext('2d');
    rewardChart = new Chart(rewardCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Recompensa Total',
                data: rewards,
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Evolução das Recompensas'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Recompensa Total'
                    }
                }
            }
        }
    });
    
    // Gráfico de arrependimento
    const regretCtx = document.getElementById('regret-chart').getContext('2d');
    regretChart = new Chart(regretCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Arrependimento Total',
                data: regrets,
                borderColor: '#e53e3e',
                backgroundColor: 'rgba(229, 62, 62, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Evolução do Arrependimento'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Arrependimento Total'
                    }
                }
            }
        }
    });
    
    // Gráfico de distribuição de recompensas
    const distributionCtx = document.getElementById('reward-distribution-chart').getContext('2d');
    rewardDistributionChart = new Chart(distributionCtx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Recompensa',
                data: rewards,
                backgroundColor: 'rgba(102, 126, 234, 0.8)',
                borderColor: '#667eea',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Distribuição de Recompensas por Iteração'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Recompensa Total'
                    }
                }
            }
        }
    });
    
    // Gráfico de comparação recompensa vs arrependimento
    const comparisonCtx = document.getElementById('reward-vs-regret-chart').getContext('2d');
    rewardVsRegretChart = new Chart(comparisonCtx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Recompensa vs Arrependimento',
                data: rewards.map((reward, index) => ({
                    x: reward,
                    y: regrets[index]
                })),
                backgroundColor: 'rgba(102, 126, 234, 0.6)',
                borderColor: '#667eea',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Correlação: Recompensa vs Arrependimento'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Recompensa Total'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Arrependimento Total'
                    }
                }
            }
        }
    });
}

// Download dos resultados
async function downloadResults() {
    if (!currentResults) {
        showMessage('Nenhum resultado para download.', 'error');
        return;
    }
    
    try {
        const response = await fetch('/download_results', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                results: currentResults
            })
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `autorl_results_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.csv`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            showMessage('Download iniciado com sucesso!', 'success');
        } else {
            showMessage('Erro ao fazer download.', 'error');
        }
    } catch (error) {
        showMessage('Erro ao fazer download.', 'error');
    }
}

// Gerar relatório completo
function downloadReport() {
    if (!currentResults) {
        showMessage('Nenhum resultado para gerar relatório.', 'error');
        return;
    }
    
    // Criar relatório em formato de texto
    const report = generateReport();
    const blob = new Blob([report], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `autorl_report_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.txt`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
    showMessage('Relatório gerado com sucesso!', 'success');
}

// Gerar relatório de texto
function generateReport() {
    const report = [];
    report.push('='.repeat(60));
    report.push('RELATÓRIO DE EXPERIMENTO AUTORL');
    report.push('='.repeat(60));
    report.push('');
    report.push(`Data: ${new Date().toLocaleString()}`);
    report.push('');
    
    // Configuração do experimento
    report.push('CONFIGURAÇÃO DO EXPERIMENTO:');
    report.push('-'.repeat(30));
    report.push(`Algoritmo: ${document.getElementById('exp-algorithm').textContent}`);
    report.push(`Otimizador: ${document.getElementById('exp-optimizer').textContent}`);
    report.push(`Gerações: ${document.getElementById('exp-generations').textContent}`);
    report.push(`Iterações: ${document.getElementById('exp-iterations').textContent}`);
    report.push('');
    
    // Estatísticas dos dados
    report.push('ESTATÍSTICAS DOS DADOS:');
    report.push('-'.repeat(25));
    report.push(`Total de Campanhas: ${document.getElementById('total-campaigns').textContent}`);
    report.push(`Total de Dias: ${document.getElementById('total-days').textContent}`);
    report.push(`Gasto Total: ${document.getElementById('total-spend').textContent}`);
    report.push(`Compras Médias: ${document.getElementById('avg-purchases').textContent}`);
    report.push('');
    
    // Resultados
    report.push('RESULTADOS:');
    report.push('-'.repeat(15));
    report.push(`Total de Execuções: ${document.getElementById('total-runs').textContent}`);
    report.push(`Recompensa Média: ${document.getElementById('avg-reward').textContent}`);
    report.push(`Melhor Recompensa: ${document.getElementById('best-reward').textContent}`);
    report.push(`Arrependimento Médio: ${document.getElementById('avg-regret').textContent}`);
    report.push('');
    
    // Detalhes das execuções
    report.push('DETALHES DAS EXECUÇÕES:');
    report.push('-'.repeat(25));
    currentResults.forEach((result, index) => {
        report.push(`Iteração ${index + 1}:`);
        report.push(`  Recompensa: ${result.total_reward.toFixed(2)}`);
        report.push(`  Arrependimento: ${result.total_regret.toFixed(2)}`);
        if (result.best_params) {
            report.push(`  Parâmetros: ${JSON.stringify(result.best_params)}`);
        }
        report.push('');
    });
    
    return report.join('\n');
}

// Mostrar mensagens
function showMessage(message, type) {
    // Remover mensagens existentes
    const existingMessages = document.querySelectorAll('.message');
    existingMessages.forEach(msg => msg.remove());
    
    // Criar nova mensagem
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    messageDiv.textContent = message;
    
    // Inserir no topo da página
    const container = document.querySelector('.container');
    container.insertBefore(messageDiv, container.firstChild);
    
    // Remover após 5 segundos
    setTimeout(() => {
        if (messageDiv.parentNode) {
            messageDiv.remove();
        }
    }, 5000);
} 