<style type="text/css">

    .tableDoc{
        width:500px; 
        vertical-align:middle;

    }
    .table{
        width:888px; 
        vertical-align:middle;

    }

    .table2{
        width:888px; 

        vertical-align:middle;


    }

    .marcoDoc1{

        width:900px !important;
        height:187px;margin-left:0px;
        margin-bottom:5px !important;
        border-color:#961C1F;
        padding-top:5px;
        padding-right:5px;
        padding-left:14px;
        -moz-border-radius: 7px; /* Firefox */
        -webkit-border-radius: 7px; /* Safari and Chrome */
        -border-radius: 7px; /* Opera 10.5+, future browsers, and now also Internet Explorer 6+ using IE-CSS3 */
    }


    .secuenciaDoc1{

        width: 130px;

    }

    .secuencias2{

        width: 250px;
        padding:0px


    }





</style>
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
    'type' => 'GET', //request typem
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
    'type' => 'GET', //request typem
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

    function updateGridMotores()
    {
<?php
echo CHtml::ajax(array(
    'type' => 'GET', //request type
    'data' => array('area' => 'js:document.getElementById("MetaDocs_UT_Proceso").value', 'equipo' => 'js:document.getElementById("MetaDocs_UT_Equipo").value'),
    'update' => '#gridMotores',
    'url' => CController::createUrl('/motores/dynamicMotores'), //url to call.
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
        updateFieldEquipo();
    };

    // función que cambia los datos de el Dropdownlist de Equipo
    function updateEquipoDropdown(data)
    {
        $('#MetaDocs_UT_Equipo').html(data.value1);
        updateFieldMotor();
    };
    
    // función que cambia los datos de el Dropdownlist de Equipo
    function updateMotorDropdown(data)
    {
        $('#MetaDocs_UT_Motor_TAG').html(data.value1);
        updateGridMotores();
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
            <td style="padding-left:10px;width:100%" >           
                <div class="row">
                    <?php echo $form->labelEx($model, 'descripcion'); ?>
                    <?php echo $form->textField($model, 'descripcion', array('size' => 60, 'maxlength' => 128, 'style' => 'width:865px;margin-right:15px', 'onchange' => 'cambiarTitulo()')); ?>
                    <?php echo $form->error($model, 'descripcion'); ?>
                </div>

                
                    </td>
                    </tr>
                    </table>
                    <table style="padding-left:10px">
                    <tr>
                        <td>
                            <b>Area:</b>
                            <?php
                            $valor = isset($modeloId) ? $modeloId->Proceso : "";
// dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
                            echo CHtml::dropDownList(
                                    'MetaDocs[UT_Area]', $valor, CHtml::listData(Estructura::model()->findAllbySql(
                                                    'SELECT DISTINCT Proceso FROM estructura ORDER BY Proceso ASC', array()), 'Proceso', 'Proceso'
                                    ), array(
                                //'onfocus' => 'updateFieldArea()',
                                'onchange' => 'updateFieldArea()',
                                'style' => 'width:427px;margin-right:10px',
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
                                                            'SELECT DISTINCT Area FROM estructura WHERE Proceso="' . $modeloId->Proceso . '" ORDER BY Area ASC' : 'SELECT DISTINCT Area FROM estructura WHERE Proceso="ELABORACION" ORDER BY Area ASC'
                                                    , array()), 'Area', 'Area'
                                    ), array(
                                //'onfocus' => 'updateFieldArea()',
                                'onchange' => 'updateFieldEquipo()',
                                 'style' => 'width:427px',
                                'class' => 'select',
                                'empty' => 'Seleccione el proceso',
                                    )
                            );
                            ?>
                        </td>
                    </tr>
                    </table>
    <table style="padding-left:10px">
                    <tr>
                        <td>
                            <b>Equipo:<br/></b>
                            <?php
                            $valor = isset($modeloId) ? $modeloId->Equipo : "";
                            // dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
                            echo CHtml::dropDownList(
                                    'MetaDocs[UT_Equipo]', $valor, CHtml::listData(Estructura::model()->findAllbySql(
                                                    isset($modeloId) ?
                                                            'SELECT Equipo FROM estructura WHERE Area="' . $modeloId->Area . '" ORDER BY Equipo ASC' : 'SELECT Equipo FROM estructura WHERE Area="FILTRACION" ORDER BY Equipo ASC', array()), 'Equipo', 'Equipo'
                                    ), array(
                                //'onfocus' => 'updateFieldEquipo()',
                                'onchange' => 'updateFieldMotor()',
                               'style' => 'width:427px;margin-right:10px',
                                'class' => 'select',
                                'empty' => 'Seleccione el equipo para filtrar el resultado',
                                    )
                            );
                            ?>
                        </td>
                        <td>
                            <b>Motor:<br/></b>
                            <?php
                            $valor = isset($modeloId) ? $modeloId->TAG : "";
// dibuja el dropDownList de Proceso, seleccionando los valores diferentes presentes en la tabla Estructura col. Proceso
                            echo CHtml::dropDownList(
                                    'MetaDocs[UT_Motor_TAG]', $valor, CHtml::listData(Motores::model()->findAllbySql(
                                                    isset($modeloId) ?
                                                            'SELECT TAG, CONCAT(TAG," - ",Motor) as Motor FROM motores WHERE Equipo="' . $modeloId->Equipo . '" ORDER BY TAG ASC' : 'SELECT TAG, CONCAT(TAG," - ",Motor) as Motor FROM motores WHERE Equipo="ANILLO DE CONTRAPRESION" ORDER BY TAG ASC', array()), 'TAG', 'Motor'
                                    ), array(
                                //'onfocus' => 'updateGraph()',
                               // 'onchange' => 'updateGridFechas()',
                                'style' => 'width:427px',
                                'class' => 'select',
                                'empty' => 'Seleccione el motor para filtar el resultado',
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
                                    'Contenido' => $this->renderPartial('_accordionOnlineMotor', array('form' => $form, 'modelA' => $modelA, 'panel' => "0"), true),
                                    'Metadatos' => $this->renderPartial('_accordionOnlineMotor', array('form' => $form, 'modelA' => $modelA, 'modelMeta' => $modelMeta, 'model' => $model, 'panel' => "1"), true),
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

