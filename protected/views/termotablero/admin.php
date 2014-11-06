<?php
//para leer el param get y con el reconfigurar los dropdown
$sufix = "";
if (isset($_GET['id'])) {
    $modeloId = Tableros::model()->findByAttributes(array("id" => $_GET['id']));
    if (isset($modeloId))
    $sufix = "?id=" . urlencode($modeloId->TAG);
}
?>
<style type="text/css">
  
  div.loading {
    background-color: #FFFFFF;
    background-image: url('/images/loading.gif');
    background-position:  100px;
    background-repeat: no-repeat;
    opacity: 1;
}
div.loading * {
    opacity: .8;
}  
    
    
    .select_admin{

        background:#ffffff;        
        border: 1px solid #DBC08F;
        -moz-border-radius:3px;
        -webkit-border-radius: 2px;
        border-radius:2px;
    }

    .back{

        width:30px      



    }


</style> 

<script type="text/javascript">
    // función que actualiza el campo de Area dependiendo del campo de proceso
    function updateFieldArea()
    {
        
        
      
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('proceso' => 'js:document.getElementById("proceso").value'),
    'url' => CController::createUrl('/motores/dynamicArea'), //url to call.
    //'update' => '#Visitas_idDoctor', //selector to update
    'success' => 'updateAreaDropdown',
    'beforeSend' => 'function(){
      $("#myDiv").addClass("loading");}',
    'complete' => 'function(){
      $("#myDiv").removeClass("loading");}',
    'dataType' => 'json')
);
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;
        return false;
    }
    
    function updateFieldEquipo()
    {
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('area' => 'js:document.getElementById("area").value'),
    'url' => CController::createUrl('/tableros/dynamicTableroDropDown'), //url to call.
    //'update' => '#Visitas_idDoctor', //selector to update
    'success' => 'updateEquipoDropdown',
    'beforeSend' => 'function(){
      $("#myDiv").addClass("loading");}',
    'complete' => 'function(){
      $("#myDiv").removeClass("loading");}',
    'dataType' => 'json')
);
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;
        return false;
    }
       
    function updateFieldTablero()
    {
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('area' => 'js:document.getElementById("area").value'),
    'url' => CController::createUrl('/termotablero/dynamicTableros'), //url to call.
    //'update' => '#divTableros', //selector to update
    'success' => 'updateTableros',
    'beforeSend' => 'function(){
      $("#myDiv").addClass("loading");}',
    'complete' => 'function(){
      $("#myDiv").removeClass("loading");}',
    'dataType' => 'json')
);
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;
        return false;
    }
    
    function updateTableros(data)
    {
        $('#TAG').html(data.value1);
        updateGrid();
    }
    

    function updateGrid()
    {
        $('#linkNuevo').html('<a href="/index.php/termotablero/create?id='+encodeURIComponent(document.getElementById("TAG").value)+'">Nuevo Informe.</a>');
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('TAG' => 'js:document.getElementById("TAG").value'),
    'update' => '#divGridTAG',
    'url' => CController::createUrl('/termotablero/dynamicGridTAG'), //url to call.
        //'update' => '#Visitas_idDoctor', //selector to update
        //'success' => 'updateContGridMotores',
        //'dataType' => 'json'
        )
);
?>
        var texto;
        //document.getElementById('Examenes_convenio').selectedIndex = conv;
        return false;
    }

    function updateGridReportesArea()
    {
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('area' => 'js:document.getElementById("area").value'),
    'update' => '#gridReportes',
    'url' => CController::createUrl('/reportes/dynamicReportesArea'), //url to call.
        //'update' => '#Visitas_idDoctor', //selector to update
        //'success' => 'updateContGridMotores',
        //'dataType' => 'json'
        )
);
?>

        return false;
    }


    // función que cambia los datos de el Dropdownlist de Area
    function updateAreaDropdown(data)
    {
        $('#area').html(data.value1);
        updateFieldEquipo();
    };

    // función que cambia los datos de el Dropdownlist de Equipo
    function updateEquipoDropdown(data)
    {
        $('#TAG').html(data.value1);
        updateGrid();
    };

    // función que cambia los datos de el Dropdownlist de Equipo
    function updateEquipoDropdownVacio(data)
    {   
        updateFieldEquipoComment();
        updateGridReportesArea();
    };
    
    // función que coloca el dropdown de Equipo  con comentario únicamente
    function updateAreaDropdownComment(data)
    {
        $('#TAG').html(data.value1);
        //updateFieldEquipo();
    };
    


</script>


<?php
$this->breadcrumbs = array(
    'Informes de Tableros Eléctricos' => array(Yii::t('app', 'index')),
    Yii::t('app', 'Gestionar'),
);

$this->menu = array(
    array('label' => Yii::t('app', 'Lista de Informes'), 'url' => array('index')),
    array('label' => Yii::t('app', 'Nuevo Informe'),
        'url' => array('create' . $sufix),
        'itemOptions' => array('id' => 'linkNuevo')),
);

Yii::app()->clientScript->registerScript('search', "
			$('.search-button').click(function(){
				$('.search-form').toggle();
				return false;
				});
			$('.search-form form').submit(function(){
				$.fn.yiiGridView.update('termotablero-grid', {
data: $(this).serialize()
});
				return false;
				});
			");
?>

<?php $this->setPageTitle(' Gestionar&nbsp; Termografia de Tableros Eléctricos'); ?>
<div name="myDiv" id="myDiv" class="forms100cb">
<form>
    <table style="width:100%; ">
        <tr>
            <td style="width:50%;">
                <b>Area:</b>
                <?php
                $valor = isset($modeloId) ? $modeloId->Proceso : "";
// dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
                echo CHtml::dropDownList(
                        'proceso', $valor, CHtml::listData(Tableros::model()->findAllbySql(
                                        'SELECT DISTINCT Proceso FROM tableros ORDER BY Proceso ASC', array()), 'Proceso', 'Proceso'
                        ), array(
                    //'onfocus' => 'updateFieldArea()',
                    'onchange' => 'updateFieldArea()',
                    'style' => 'width:100%;',
                    'class' => 'select'
                        )
                );
                ?>
                <!-- an la app original era:SELECT DISTINCT Area , Indicativo FROM Estructura WHERE (Proceso=@Proceso) ORDER BY Indicativo ASC -->
            </td>
            <td style="width:50%;">
                <b>Proceso:</b>
                <?php
                $valor = isset($modeloId) ? $modeloId->Area : "";
                // dibuja el dropDownList de Area, dependiendo del proceso selecccionado
                echo CHtml::dropDownList(
                        'area', $valor, CHtml::listData(Tableros::model()->findAllbySql(
                                        isset($modeloId) ?
                                                'SELECT DISTINCT Area FROM tableros WHERE Proceso="' . $modeloId->Proceso . '" ORDER BY Area ASC' : 'SELECT DISTINCT Area FROM tableros WHERE Proceso="ELABORACION" ORDER BY Area ASC', array()), 'Area', 'Area'
                        ), array(
                    'onchange' => 'updateFieldTablero()',
                    'style' => 'width:100%;',
                    'class' => 'select',
                    'empty' => 'Seleccione un proceso',
                        )
                );
                ?>
            </td>
        </tr>
    </table>

    <table style="width:100%; ">
        <tr>
            <td>
                <b>Tablero:<br/></b>
                <div id="divTableros" name="divTableros">
                    <?php
                    $valor = isset($modeloId) ? $modeloId->TAG : "";
                    // dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
                    echo CHtml::dropDownList(
                            'TAG', $valor, CHtml::listData(Tableros::model()->findAllbySql(
                                            isset($modeloId) ?
                                                    'SELECT TAG,Tablero FROM tableros WHERE Area="' . $modeloId->Area . '" ORDER BY Tablero ASC' : 'SELECT TAG,Tablero FROM tableros WHERE Area="FILTRACION" ORDER BY Tablero ASC', array()), 'TAG', 'Tablero'
                            )
                            , array(
                        //'onfocus' => 'updateGrid()',
                        'onchange' => 'updateGrid()',
                        'style' => 'width:100%;',
                        'class' => 'select',
                        'empty' => 'Seleccione un tablero',
                            )
                    );
                    ?>
                </div>
            </td>
        </tr>
    </table>
</form>
</div>






<div id="divGridTAG" name="divGridTAG">
    <?php
    if (isset($modeloId)) {
        echo "<script language=javascript>updateGrid()</script>";
    } else {
        
  /*
        $this->widget('zii.widgets.grid.CGridView', array(
            'id' => 'clinicas-grid',
            'dataProvider' => $model->search(),
            //  'filter' => $model,
            'cssFile' => '/themes/gridview/styles.css',     'template'=> '{items}{pager}{summary}',     'summaryText'=>'Resultados del {start} al {end} de {count} encontrados',
            'columns' => array(
                'Fecha',
                'TAG',
                array(// related city displayed as a link
                    'header' => 'Reporte',
                    'type' => 'raw',
                    'value' => 'CHtml::link("Ver reporte", (isset($data->Path))?$data->Path:"No existe", array("class" => "back"))'
                ),
                'Analista',
                array(
                    'class' => 'CButtonColumn',
                ),
            ),
        ));
  */  }
    ?>
</div>

