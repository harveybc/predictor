<style>

    .tableDoc{
        width:888px; 
        vertical-align:middle;

    }

    .tableDoc2{
        width:850px;
        padding-left:30px

    }


    table{
        width:888px; 
        vertical-align:middle;
        margin:0px;
        padding:0px;

    }

    .table2{
        width:888px; 

        vertical-align:middle;


    }
.marcoDoc1{

        width:900px !important;
        height:140px;margin-left:0px;
        margin-bottom:5px !important;
        border-color:#961C1F;
        padding-top:5px;
        padding-right:5px;
        padding-left:14px;
        -moz-border-radius: 7px; /* Firefox */
        -webkit-border-radius: 7px; /* Safari and Chrome */
        -border-radius: 7px; /* Opera 10.5+, future browsers, and now also Internet Explorer 6+ using IE-CSS3 */
    }
    .marcoDoc{

        width:888px !important;

        margin-bottom:5px !important;
        border-color:#961C1F;
        padding-top:5px;
        padding-right:5px;
        padding-left:14px;
        -moz-border-radius: 7px; /* Firefox */
        -webkit-border-radius: 7px; /* Safari and Chrome */
        -border-radius: 7px; /* Opera 10.5+, future browsers, and now also Internet Explorer 6+ using IE-CSS3 */
    }

    .marcoMeta {

        width:888px !important;

        margin:0px !important;
        border-color:#961C1F;
        padding:0px;
        -moz-border-radius: 7px; /* Firefox */
        -webkit-border-radius: 7px; /* Safari and Chrome */
        -border-radius: 7px; /* Opera 10.5+, future browsers, and now also Internet Explorer 6+ using IE-CSS3 */

    }

    .marcoMeta2 {

        width:888px !important;

        margin:0px !important;
        border-color:#961C1F;
        padding:0px;
        -moz-border-radius: 7px; /* Firefox */
        -webkit-border-radius: 7px; /* Safari and Chrome */
        -border-radius: 7px; /* Opera 10.5+, future browsers, and now also Internet Explorer 6+ using IE-CSS3 */

    }

    .secuenciaDoc{

        width: 200px;

    }

    .secuencias{

        width: 150px;

    }

    .secuencias2{

        width: 303px;

    }

    .select1{
        width:300px;
        background:#ffffff;        
        border: 1px solid #DBC08F;
        -moz-border-radius:3px;
        -webkit-border-radius: 2px;
        border-radius:3px;
    }
    td{

        padding:0px;
        margin:0px;

    }


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

    .select{

        width: 30px;
        padding-left:2px;
        background:#ffffff;
        background-color:#ffffff;
        border: 1px solid #DBC08F;
        -moz-border-radius:3px;
        -webkit-border-radius: 3px;
        border-radius:3px;


    }
    .back{


        padding-left:605px


    }




</style>

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

    .select{

        background:#ffffff;        
        border: 1px solid #DBC08F;
        -moz-border-radius:3px;
        -webkit-border-radius: 2px;
        border-radius:2px;


    }



</style>


<script type="text/javascript">
    
    function updateGridEstructura()
    {
        
        
      
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('area' => 'js:document.getElementById("MetaDocs_UT_area").value'),
    'url' => CController::createUrl('/estructura/dynamicEstructura'), //url to call.
    'update' => '#divEstructura', //selector to update
        )
);
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;
        return false;
    }
    // función que actualiza el campo de Area dependiendo del campo de proceso
    function updateFieldArea()
    {
        
        
      
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('proceso' => 'js:document.getElementById("MetaDocs_UT_Area").value'),
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
    'data' => array('area' => 'js:document.getElementById("MetaDocs_UT_Proceso").value'),
    'url' => CController::createUrl('/motores/dynamicEquipo'), //url to call.
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

    function updateFieldMotor()
    {
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('equipo' => 'js:document.getElementById("MetaDocs_UT_Equipo").value'),
    'url' => CController::createUrl('/motores/dynamicFMotor'), //url to call.
    //'update' => '#Visitas_idDoctor', //selector to update
    'success' => 'updateMotorDropdown',
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

    function updateFieldEquipoVacio()
    {
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('area' => 'js:document.getElementById("area").value'),
    'url' => CController::createUrl('/motores/dynamicEquipoVacio'), //url to call.
    //'update' => '#Visitas_idDoctor', //selector to update
    'success' => 'updateEquipoDropdownVacio',
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
    
    function updateFieldEquipoComment()
    {
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('area' => 'js:document.getElementById("area").value'),
    'url' => CController::createUrl('/motores/dynamicEquipoVacio'), //url to call.
    //'update' => '#Visitas_idDoctor', //selector to update
    'success' => 'updateAreaDropdownComment',
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

    function updateGridReportes()
    {
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('area' => 'js:document.getElementById("area").value', 'equipo' => 'js:document.getElementById("equipo").value'),
    'update' => '#divGridReportes',
    'url' => CController::createUrl('/reportes/dynamicReportes'), //url to call.
        //'update' => '#Visitas_idDoctor', //selector to update
        //'success' => 'updateContGridMotores',
        //'dataType' => 'json'
        )
);
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;
        return false;
    }

    function updateGridReportesArea()
    {
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('area' => 'js:document.getElementById("area").value'),
    'update' => '#divGridReportes',
    'url' => CController::createUrl('/reportes/dynamicReportesArea'), //url to call.
        //'update' => '#Visitas_idDoctor', //selector to update
        //'success' => 'updateContGridMotores',
        //'dataType' => 'json'
        )
);
?>
        //document.getElementById('Examenes_convenio').selectedIndex = conv;
        return false;
    }


    // función que cambia los datos de el Dropdownlist de Area
    function updateAreaDropdown(data)
    {
        $('#MetaDocs_UT_Proceso').html(data.value1);
        //  $("#area").trigger("liszt:updated");
        // $("#equipo").trigger("liszt:updated");
        // updateGridEstructura();
        updateFieldEquipo();
        //updateGridTableros();
        
    };

    // función que cambia los datos de el Dropdownlist de Equipo
    function updateEquipoDropdown(data)
    {
        $('#MetaDocs_UT_Equipo').html(data.value1);
        //updateGridReportes();
        //  $("#equipo").trigger("liszt:updated");
        // $("#motor").trigger("liszt:updated");
    };

    // función que cambia los datos de el Dropdownlist de Equipo
    function updateMotorDropdown(data)
    {
        $('#MetaDocs_UT_Motor_TAG').html(data.value1);
        //updateGridReportes();
        //  $("#equipo").trigger("liszt:updated");
        // $("#motor").trigger("liszt:updated");
    };

    // función que cambia los datos de el Dropdownlist de Equipo
    function updateEquipoDropdownVacio(data)
    {   
        updateFieldEquipoComment();
        updateGridReportesArea();
        //  $("#equipo").trigger("liszt:updated");
    };
    
    // función que coloca el dropdown de Equipo  con comentario únicamente
    function updateAreaDropdownComment(data)
    {
        $('#equipo').html(data.value1);
        //   $("#motor").trigger("liszt:updated");
        //updateFieldEquipo();
    };
    


</script>

<script>
    function cambiarTitulo(){
        
        var conv=$('#Documentos_descripcion').attr('value');
        $('#MetaDocs_titulo').attr('value',conv);
    }
</script>



<?php echo $form->errorSummary(array($model, $modelMeta, $modelA)); ?>
<br/>
<fieldset class="marcoDoc1">

    <legend style="color:#961C1F"><b>Datos Básicos</b></legend>

    <table class="tableDoc">
        <tr>
            <td style="padding-left:10px" >           
                <div class="row">
                    <?php echo $form->labelEx($model, 'descripcion'); ?>
                    <?php echo $form->textField($model, 'descripcion', array('size' => 60, 'maxlength' => 128, 'style' => 'width:860px;', 'onchange' => 'cambiarTitulo()')); ?>
                    <?php echo $form->error($model, 'descripcion'); ?>
                </div>


            </td>
            </tr>
            </table>
    <table style="padding-left:10px;width:900px;margin-left:0px;margin-bottom:5px !important; ">
            <tr>
                <td >
                    <b>Area:</b>
                    <?php
                    $valor = isset($modeloId) ? $modeloId->Proceso : "";
// dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
                    echo CHtml::dropDownList(
                            'MetaDocs[UT_Area]', $valor, CHtml::listData(Estructura::model()->findAllbySql(
                                            'SELECT DISTINCT Proceso FROM estructura ORDER BY Proceso ASC', array()), 'Proceso', 'Proceso'
                            ), array(
                        //'onfocus' => 'updateFieldArea()',
                        'onchange' => 'updateFieldArea();',
                        'style' => 'width:170px;margin-right:15px',
                        'class' => 'select'
                            )
                    );
                    ?>
                    <!-- an la app original era:SELECT DISTINCT Area , Indicativo FROM Estructura WHERE (Proceso=@Proceso) ORDER BY Indicativo ASC -->
                </td>
                <td> 
                    <b>Proceso:</b>
                    <?php
                    $valor = isset($modeloId) ? $modeloId->Area : "";
// dibuja el dropDownList de Area, dependiendo del proceso selecccionado
                    echo CHtml::dropDownList(
                            'MetaDocs[UT_Proceso]', $valor, CHtml::listData(Estructura::model()->findAllbySql(
                                            isset($modeloId) ?
                                                    'SELECT DISTINCT Area FROM estructura WHERE Proceso="' . $modeloId->Proceso . '" ORDER BY Area ASC' : 'SELECT DISTINCT Area FROM estructura WHERE Proceso="ELABORACION" ORDER BY Area ASC', array()), 'Area', 'Area'
                            ), array(
                        'onchange' => 'updateFieldEquipo();',
                        'style' => 'width:170px;margin-right:15px',
                        'class' => 'select',
                        'empty' => 'Seleccione el proceso',
                            )
                    );
                    ?>
                </td>
        
                <td>
                    <b>Equipo:<br/></b>
                    <?php
                        if (isset($modeloId))
                            {
                            $model->Equipo = $modeloId->Equipo;
                            }
                        // dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
                        
                        echo CHtml::activeDropDownList($modelMeta,
                                                       'UT_Equipo',
                                                       CHtml::listData(Estructura::model()->findAllbySql(
                                    'SELECT Equipo FROM estructura WHERE Area="' .(isset($modeloId)? $modeloId->Area:"FILTRACION" ). '" ORDER BY Equipo ASC',
                                                                                                    array ()),
                                                                                                    'Equipo',
                                                                                                    'Equipo'
                            ),
                                                       array (
                            //'onfocus'  => 'updateFieldEquipo()',
                            //'onchange' => 'updateGridReportes()',
                            'style' => 'width:500px;margin-right:30px',
                            'class' => 'select',
                            'empty' => 'Seleccione el equipo',
                            )
                        );
                    ?>
                </td>
            </tr>
        </table> 

</fieldset>

<?php
//echo $form->textField($modelA,'contenido');

$this->widget('zii.widgets.jui.CJuiAccordion', array(
    'panels' => array(
        'Contenido' => $this->renderPartial('_accordionOnlineEquipo', array('form' => $form, 'modelA' => $modelA, 'panel' => "0"), true),
        'Metadatos' => $this->renderPartial('_accordionOnlineEquipo', array('form' => $form, 'modelA' => $modelA, 'modelMeta' => $modelMeta, 'model' => $model, 'panel' => "1"), true),
    ),
    // additional javascript options for the accordion plugin
    'options' => array(
        'animated' => 'bounceslide',
    ),
    'htmlOptions' => array('style' => 'width:920px;margin:0px;padding-right:1px'),
    'themeUrl' => '/themes',
    'theme' => 'acordeonSgdoc',
));
?>

