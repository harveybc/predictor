<style type="text/css">

    .tableDoc{
        width:888px; 
        vertical-align:middle;

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

        width: 150px;

    }

    .secuencias{

        width: 150px;

    }
    .secuencias2{

        width: 250px;
        padding:0px
       

    }
    td{

            padding:0px;
            margin:0px;

        }



</style>

<script>
    function cambiarTitulo(){
        
        var conv=$('#Documentos_descripcion').attr('value');
        $('#MetaDocs_titulo').attr('value',conv);
    }
</script>


<br/>

<?php
    if (0+$panel==0)
    {
?>


<table class="tableDoc">
        <tr>
            <td >           
                <div class="row">
                    <?php echo $form->labelEx($modelMeta, 'titulo'); ?>
                    <?php echo $form->textField($modelMeta, 'titulo', array('size' => 60, 'maxlength' => 128, 'style' => 'width:250px')); ?>
                    <?php echo $form->error($modelMeta, 'titulo'); ?>
                </div>
            </td>
             <td  style="padding-left:25px">
                <label for="UbicacionTec">Ubicación Técnica</label><?php
                $this->widget('application.components.Relation', array(
                    'model' => $modelMeta,
                    'relation' => 'ubicacionT0',
                    'fields' => 'descripcion',
                    'allowEmpty' => true,
                    'style' => 'dropdownlist',
                    'htmlOptions' => array(
                        'class' => 'secuencias2',)
                        )
                );
            ?>
            </td>
            <td>
                <div class="row">
                    <?php echo $form->labelEx($modelMeta, 'autores'); ?>
                    <?php echo $form->textField($modelMeta, 'autores', array('size' => 60, 'maxlength' => 256, 'style' => 'width:250px')); ?>
                    <?php echo $form->error($modelMeta, 'autores'); ?>
                </div>

            </td>
            </tr>
            </table>
<table>
            <tr>
   
             <td>
                <div class="row">
                    <?php echo $form->labelEx($modelMeta, 'modulo'); ?>
                    <?php echo $form->textField($modelMeta, 'modulo', array('style' => 'width:20px;padding-left:20px')); ?>
                    <?php echo $form->error($modelMeta, 'modulo'); ?>
                </div>
            </td>
            <td>

                <div class="row">
                    <?php echo $form->labelEx($modelMeta, 'columna'); ?>
                    <?php echo $form->textField($modelMeta, 'columna', array('style' => 'width:20px')); ?>
                    <?php echo $form->error($modelMeta, 'columna'); ?>
                </div>


            </td>
            <td style="padding-right:30px">
                <div class="row">
                    <?php echo $form->labelEx($modelMeta, 'fila'); ?>
                    <?php echo $form->textField($modelMeta, 'fila', array('style' => 'width:20px')); ?>
                    <?php echo $form->error($modelMeta, 'fila'); ?>
                </div>
            </td>
           
         
<td>
                <div class="row">
                    <?php echo $form->labelEx($modelMeta, 'disponibles'); ?>
                    <?php echo $form->textField($modelMeta, 'disponibles', array('style' => 'width:100px')); ?>
                    <?php echo $form->error($modelMeta, 'disponibles'); ?>
                </div>
            </td>
          
            <td>
                <div class="row">
                    <?php echo $form->labelEx($modelMeta, 'existencias'); ?>
                    <?php echo $form->textField($modelMeta, 'existencias', array('style' => 'width:100px')); ?>
                    <?php echo $form->error($modelMeta, 'existencias'); ?>
                </div>
            </td>
            
            <td>
                <div class="row">
                    <?php echo $form->labelEx($modelMeta, 'ISBN'); ?>
                    <?php echo $form->textField($modelMeta, 'ISBN', array('size' => 32, 'maxlength' => 32, 'style' => 'width:100px')); ?>
                    <?php echo $form->error($modelMeta, 'ISBN'); ?>
                </div>
            </td>
            <td>
                <div class="row">
                    <?php echo $form->labelEx($modelMeta, 'EAN13'); ?>
                    <?php echo $form->textField($modelMeta, 'EAN13', array('size' => 32, 'maxlength' => 32, 'style' => 'width:100px')); ?>
                    <?php echo $form->error($modelMeta, 'EAN13'); ?>
                </div>
            </td>
            



            
        </tr>
    </table>

<?php
    }
   
    if (0+$panel==1)
    {
     ?>

 <table class="table">
        <tr>

            <td>
                  
               
                <div class="row">
                    <?php echo $form->labelEx($modelMeta, 'descripcion'); ?>
                    <?php echo $form->textArea($modelMeta, 'descripcion', array('size' => 60, 'maxlength' => 256, 'style' => 'width:150px')); ?>
                    <?php echo $form->error($modelMeta, 'descripcion'); ?>
                </div>

            </td>

            <td>
                <div class="row">
                    <?php echo $form->labelEx($modelMeta, 'numPedido'); ?>
                    <?php echo $form->textField($modelMeta, 'numPedido', array('size' => 60, 'maxlength' => 64, 'style' => 'width:150px')); ?>
                    <?php echo $form->error($modelMeta, 'numPedido'); ?>
                </div>
            </td>
            <td>
                <div class="row">
                    <?php echo $form->labelEx($modelMeta, 'numComision'); ?>
                    <?php echo $form->textField($modelMeta, 'numComision', array('size' => 60, 'maxlength' => 64, 'style' => 'width:150px')); ?>
                    <?php echo $form->error($modelMeta, 'numComision'); ?>
                </div>                
            </td>
            <td>
                 <?php echo $form->labelEx($modelMeta, 'fechaRecepcion'); ?>
                <?php
//Fecha inicial
                $today = date("Y-m-d H:i:s");
                if (isset($modelMeta->fechaRecepcion))
                    $today = $modelMeta->fechaRecepcion;
                else
                    $modelMeta->fechaRecepcion= $today;
//fin Fecha inicial
                Yii::import('application.extensions.CJuiDateTimePicker.CJuiDateTimePicker');
                $this->widget('CJuiDateTimePicker', array(
                    'model' => $modelMeta, //Model object
                    'attribute' => 'fechaRecepcion', //attribute name
                    'mode' => 'datetime', //use "time","date" or "datetime" (default)
                    'language' => 'es',
                    //    'value' =>$today,
                    'themeUrl' => '/themes',
                    'theme' => 'calendariocbm',
                    'htmlOptions'=>array('style'=>'width:140px;'),
                    'options' => array(
                        'dateFormat' => 'yy-mm-dd',
                        'showButtonPanel' => true,
                        "yearRange" => '1995:2070',
                        'changeYear' => true,
                        'buttonImage' => '/images/calendar.png',
                        'showOn' => "both",
                        'buttonText' => "Seleccione la fecha",
                        'buttonImageOnly' => true,
                    ) // jquery plugin options
                ));
                ?>
            </td> 
            

            
        </tr>
    
        <tr>
             <td>
                <label for="Fabricantes">Fabricante</label><?php
                $this->widget('application.components.Relation', array(
                    'model' => $modelMeta,
                    'relation' => 'fabricante0',
                    'fields' => 'descripcion',
                    'allowEmpty' => false,
                    'style' => 'dropdownlist',
                    'htmlOptions' => array(
                        'class' => 'secuencias',)
                        )
                );
                    ?>
            </td>
            

<td>
                <label for="Idiomas">Idioma</label><?php
                $this->widget('application.components.Relation', array(
                    'model' => $modelMeta,
                    'relation' => 'idioma0',
                    'fields' => 'descripcion',
                    'allowEmpty' => false,
                    'style' => 'dropdownlist',
                    'htmlOptions' => array(
                        'class' => 'secuencias',)
                        )
                );
            ?>
            </td>
            <td>
                <label for="Cervecerias">Cerveceria </label><?php
                    $this->widget('application.components.Relation', array(
                        'model' => $modelMeta,
                        'relation' => 'cerveceria0',
                        'fields' => 'descripcion',
                        'allowEmpty' => true,
                        'style' => 'dropdownlist',
                        'htmlOptions' => array(
                            'class' => 'secuencias',)
                            )
                    );
                    ?>

            </td>

           
 
           
            <td>
                 <?php echo $form->hiddenField($modelMeta, 'documento', array('value' => 1)); ?>
                
                
                <label for="TipoContenidos">Tipo de Contenido</label><?php
            $this->widget('application.components.Relation', array(
                'model' => $modelMeta,
                'relation' => 'tipoContenido0',
                'fields' => 'descripcion',
                'allowEmpty' => false,
                'style' => 'dropdownlist',
                'htmlOptions' => array(
                    'class' => 'secuencias',)
                    )
            );
            ?>

            </td>
           
            
            </tr>
         
            <tr>
   <td>
                <label for="Medios">Medio</label><?php
                $this->widget('application.components.Relation', array(
                    'model' => $modelMeta,
                    'relation' => 'medio0',
                    'fields' => 'descripcion',
                    'allowEmpty' => false,
                    'style' => 'dropdownlist',
                    'htmlOptions' => array(
                        'class' => 'secuencias',)
                        )
                );
            ?>

            </td>
            
             
            
            
            
             <td>
                <div class="row">
                    <?php echo $form->labelEx($modelMeta, 'version'); ?>
                    <?php echo $form->textField($modelMeta, 'version', array('size' => 60, 'maxlength' => 64, 'style' => 'width:150px')); ?>
                    <?php echo $form->error($modelMeta, 'version'); ?>
                </div>
            </td>


                

    <td>
                <label for="Secuencias">Secuencia</label><?php
                    $this->widget('application.components.Relation', array(
                        'model' => $modelMeta,
                        'relation' => 'secuencia0',
                        'fields' => 'descripcion',
                        'allowEmpty' => false,
                        'style' => 'dropdownlist',
                        'htmlOptions' => array(
                            'class' => 'secuenciaDoc',)
                            )
                    );
                    ?>

            </td>
            <td>
                <label for="OrdenSecuencias">Orden de secuencia</label><?php
                $this->widget('application.components.Relation', array(
                    'model' => $modelMeta,
                    'relation' => 'ordenSecuencia0',
                    'fields' => 'posicion',
                    'allowEmpty' => true,
                    'style' => 'dropdownlist',
                    'htmlOptions' => array(
                        'class' => 'secuenciaDoc',)
                        )
                );
                    ?>
            </td>
   
            
            
        </tr>
               </table>



<?php
    }
   
    if (0+$panel==2)
    {
     ?>
<p class="note">Campos con<span class="required">*</span> son necesarios.</p>



    <table>
        <tr>
            <td>
 <?php echo $form->labelEx($modelMeta, 'fechaCreacion'); ?>
                <?php
//Fecha inicial
                $today = date("Y-m-d H:i:s");
                if (isset($modelMeta->fechaCreacion))
                    $today = $modelMeta->fechaCreacion;
                else
                    $modelMeta->fechaCreacion= $today;
//fin Fecha inicial
                Yii::import('application.extensions.CJuiDateTimePicker.CJuiDateTimePicker');
                $this->widget('CJuiDateTimePicker', array(
                    'model' => $modelMeta, //Model object
                    'attribute' => 'fechaCreacion', //attribute name
                    'mode' => 'datetime', //use "time","date" or "datetime" (default)
                    'language' => 'es',
                    //    'value' =>$today,
                    'themeUrl' => '/themes',
                    'theme' => 'calendariocbm',
                    'htmlOptions'=>array('style'=>'width:140px;'),
                    'options' => array(
                        'dateFormat' => 'yy-mm-dd',
                        'showButtonPanel' => true,
                        "yearRange" => '1995:2070',
                        'changeYear' => true,
                        'buttonImage' => '/images/calendar.png',
                        'showOn' => "both",
                        'buttonText' => "Seleccione la fecha",
                        'buttonImageOnly' => true,
                    ) // jquery plugin options
                ));
                ?>
            </td>
            <td>
                <label for="Usuarios">Documento subido por Usuario</label><?php
                    $this->widget('application.components.Relation', array(
                        'model' => $modelMeta,
                        'relation' => 'usuario0',
                        'fields' => 'nombre',
                        'allowEmpty' => false,
                        'style' => 'dropdownlist',
                        'htmlOptions' => array(
                            'class' => 'secuencias',)
                            )
                    );
                    ?>
            </td>

            <td>
                <div class="row">
                    <?php echo $form->labelEx($modelMeta, 'revisado'); ?>
                    <?php echo $form->checkBox($modelMeta, 'revisado'); ?>
                    <?php echo $form->error($modelMeta, 'revisado'); ?>
                </div>


            </td>

            <td>
                <?php echo $form->labelEx($modelMeta, 'fechaRevisado'); ?>
                <?php
//Fecha inicial
                $today = date("Y-m-d H:i:s");
                if (isset($modelMeta->fechaRevisado))
                    $today = $modelMeta->fechaRevisado;
             //   else
                 //   $modelMeta->fechaRevisado= $today;
//fin Fecha inicial
                Yii::import('application.extensions.CJuiDateTimePicker.CJuiDateTimePicker');
                $this->widget('CJuiDateTimePicker', array(
                    'model' => $modelMeta, //Model object
                    'attribute' => 'fechaRevisado', //attribute name
                    'mode' => 'datetime', //use "time","date" or "datetime" (default)
                    'language' => 'es',
                    //    'value' =>$today,
                    'themeUrl' => '/themes',
                    'theme' => 'calendariocbm',
                    'htmlOptions'=>array('style'=>'width:140px;'),
                    'options' => array(
                        'dateFormat' => 'yy-mm-dd',
                        'showButtonPanel' => true,
                        "yearRange" => '1995:2070',
                        'changeYear' => true,
                        'buttonImage' => '/images/calendar.png',
                        'showOn' => "both",
                        'buttonText' => "Seleccione la fecha",
                        'buttonImageOnly' => true,
                    ) // jquery plugin options
                ));
                ?>
            </td>

            <td>
                <label for="Usuarios">Fué revisado por</label><?php
                    $this->widget('application.components.Relation', array(
                        'model' => $modelMeta,
                        'relation' => 'userRevisado0',
                        'fields' => 'nombre',
                        'allowEmpty' => false,
                        'style' => 'dropdownlist',
                        'htmlOptions' => array(
                            'class' => 'secuencias',)
                            )
                    );
                    ?>
            </td>
        </tr>
    </table>
<?php   
    }
?>

