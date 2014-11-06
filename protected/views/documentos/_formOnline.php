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

        width:755px !important;
        height:90px;margin-left:0px;
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
            <td  >           
                <div class="row">
                    <?php echo $form->labelEx($model, 'descripcion'); ?>
                    <?php echo $form->textField($model, 'descripcion', array('size' => 60, 'maxlength' => 128, 'style' => 'width:300px', 'onchange' => 'cambiarTitulo()')); ?>
                    <?php echo $form->error($model, 'descripcion'); ?>
                </div>
            </td>

             <td>
                <label for="UbicacionTec">Ubicación Técnica</label><?php
                $this->widget('application.components.Relation', array(
                    'model' => $modelMeta,
                    'relation' => 'ubicacionT0',
                    'fields' => 'codigoSAP',
                    'allowEmpty' => true,
                    'style' => 'dropdownlist',
                    'htmlOptions' => array(
                        'class' => 'secuencias2',)
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
        'Contenido' => $this->renderPartial('_accordionV', array('form'=>$form ,'modelA' => $modelA, 'panel' => "0"), true),
        'Metadatos' => $this->renderPartial('_accordionV', array('form'=>$form ,'modelA' => $modelA,'modelMeta' => $modelMeta,'model' => $model, 'panel' => "1"), true),
        
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

